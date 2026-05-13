/**
 * @file fourier.c
 * @brief Bandlimited Fourier waveform helpers (BLIT / PolyBLEP / miniBLEP).
 *
 * Mapping to generators:
 *  - Saw: integrate BLIT.
 *  - Square: difference of phase-shifted BLITs, then integrate.
 *  - Triangle: double integrate (square) or apply BLAMP correction.
 *  - Real-time: PolyBLEP/PolyBLAMP; higher stopband: miniBLEP/miniBLAMP.
 *
 * Phase arguments are radians for BLIT/Dirichlet APIs and normalized [0, 1]
 * phase for BLEP/BLAMP correction APIs. Complex variants preserve the same real
 * control parameters and return ABI-stable complex structures.
 */

#include "oakfield/math/fourier.h"

#include <float.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define SIM_FOURIER_TWO_PI (2.0 * M_PI)
#define SIM_FOURIER_MINIBLEP_TABLE_SIZE 257
#define SIM_FOURIER_MINIBLEP_AREA (-0.5017089863328562)

/**
 * @brief Wrap a phase into the principal interval around zero.
 *
 * @param phase Phase in radians.
 * @return Phase wrapped approximately into [-pi, pi].
 */
static inline double sim_fourier_wrap_two_pi(double phase) {
    double wrapped = phase;
    if (wrapped > SIM_FOURIER_TWO_PI || wrapped < -SIM_FOURIER_TWO_PI) {
        wrapped = fmod(wrapped, SIM_FOURIER_TWO_PI);
    }
    if (wrapped > M_PI) {
        wrapped -= SIM_FOURIER_TWO_PI;
    } else if (wrapped < -M_PI) {
        wrapped += SIM_FOURIER_TWO_PI;
    }
    return wrapped;
}

/**
 * @brief Clamp a normalized phase to [0, 1] unless fast-path validation is enabled.
 *
 * @param x Candidate normalized phase.
 * @return Clamped or asserted normalized phase.
 */
static inline double sim_fourier_clamp01(double x) {
#if SIM_FOURIER_ASSUME_VALID
#ifndef NDEBUG
    assert(x >= -1e-6 && x <= 1.0 + 1e-6);
#endif
    return x;
#else
    if (x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
#endif
}

/**
 * @brief Add a scaled complex term to a complex accumulator.
 *
 * @param acc Existing accumulator.
 * @param term Complex term to add.
 * @param scale Real scale applied to `term`.
 * @return `acc + scale * term`.
 */
static inline SimComplexDouble
sim_fourier_complex_scale_add(SimComplexDouble acc, SimComplexDouble term, double scale) {
    SimComplexDouble out = { acc.re + scale * term.re, acc.im + scale * term.im };
    return out;
}

/**
 * @brief Scale a complex value by a real factor.
 *
 * @param z Complex value.
 * @param scale Real scale factor.
 * @return `scale * z`.
 */
static inline SimComplexDouble sim_fourier_complex_scale(SimComplexDouble z, double scale) {
    SimComplexDouble out = { z.re * scale, z.im * scale };
    return out;
}

/**
 * @brief Evaluate the normalized sinc function with a small-argument series.
 *
 * @param x Real argument.
 * @return `sin(x) / x`, with the removable singularity handled near zero.
 */
static inline double sim_fourier_sinc(double x) {
    double ax = fabs(x);
    if (ax < 1e-8) {
        double x2 = x * x;
        double x4 = x2 * x2;
        return 1.0 - x2 / 6.0 + x4 / 120.0;
    }
    return sin(x) / x;
}

/**
 * @brief Convert a positive-harmonic count to an odd Dirichlet length.
 *
 * @param harmonic_count Number of positive harmonics requested.
 * @return Odd length `2N + 1`, clamped to at least one.
 */
static inline int sim_fourier_dirichlet_length(int harmonic_count) {
    if (harmonic_count < 0) {
        harmonic_count = 0;
    }
    /* M = 2N + 1 ensures an odd Dirichlet length for 2π periodicity. */
    int length = 2 * harmonic_count + 1;
    if ((length & 1) == 0) {
        length += 1;
    }
    return (length > 0) ? length : 1;
}

/**
 * @brief Wrap an integrated bipolar waveform sample back into [-1, 1).
 *
 * @param x Unwrapped bipolar value.
 * @return Centered wrapped value.
 */
static inline double sim_fourier_center_bipolar(double x) {
    double wrapped = fmod(x + 1.0, 2.0);
    if (wrapped < 0.0) {
        wrapped += 2.0;
    }
    return wrapped - 1.0;
}

/*------------------------------------------------------------------------------
 * miniBLEP / miniBLAMP tables (Blackman-windowed sinc, 8 zero crossings)
 *------------------------------------------------------------------------------*/
static const double g_sim_miniblep_table[257] = {
    -1,
    -1.000000006643911,
    -1.0000001044337226,
    -1.0000005199765749,
    -1.0000016047666946,
    -1.0000037936503532,
    -1.00000754811658,
    -1.0000132882575723,
    -1.0000213183239766,
    -1.0000317516313357,
    -1.0000444410944958,
    -1.000058921828233,
    -1.0000743720214944,
    -1.0000895976538697,
    -1.0001030455805606,
    -1.0001128480918984,
    -1.0001169003030836,
    -1.0001129697186719,
    -1.0000988351340565,
    -1.0000724497901592,
    -1.0000321215090069,
    -0.99997670053720944,
    -0.99990576414504129,
    -0.99981978580072339,
    -0.99972027608154224,
    -0.99960988249595528,
    -0.99949243614820193,
    -0.99937293472061262,
    -0.9992574535814801,
    -0.9991529799075396,
    -0.99906716845360599,
    -0.99900802187553595,
    -0.99898350314045281,
    -0.99900109232520962,
    -0.99906730476412497,
    -0.99918719179262672,
    -0.99936384896815134,
    -0.99959795936229279,
    -0.99988740105812213,
    -1.000226948139503,
    -1.0006080930625436,
    -1.0010190152563943,
    -1.0014447160926314,
    -1.0018673340575206,
    -1.0022666462195018,
    -1.0026207531578608,
    -1.0029069347487012,
    -1.0031026540108228,
    -1.0031866760812558,
    -1.0031402598474315,
    -1.0029483713613943,
    -1.0026008614472075,
    -1.0020935453979678,
    -1.001429120792269,
    -1.0006178605977298,
    -0.99967802310831877,
    -0.99863592797780676,
    -0.99752565859958109,
    -0.99638836510916673,
    -0.99527115894325913,
    -0.99422560860393527,
    -0.99330586632268814,
    -0.99256647584081015,
    -0.99205993156826633,
    -0.99183407794248535,
    -0.99192945385870102,
    -0.99237669959813124,
    -0.9931941518403008,
    -0.99438575534983664,
    -0.99593941719980805,
    -0.99782592058364505,
    -0.99999850028664128,
    -1.0023931609304924,
    -1.0049297926595393,
    -1.0075141077873477,
    -1.0100403871252568,
    -1.0123949875746185,
    -1.014460524587206,
    -1.0161206059343095,
    -1.0172649586030671,
    -1.0177947602908048,
    -1.0176279625509654,
    -1.0167043756576142,
    -1.0149902769653909,
    -1.0124823059081307,
    -1.0092104203949683,
    -1.0052397114049041,
    -1.0006709047798441,
    -0.99563942084209611,
    -0.99031291233229601,
    -0.98488725766958318,
    -0.97958104769268683,
    -0.97462866755438438,
    -0.97027213879407226,
    -0.96675194715954915,
    -0.96429713682788265,
    -0.96311499872165962,
    -0.96338071727805863,
    -0.96522736427323819,
    -0.96873663852576497,
    -0.97393074540186353,
    -0.98076578950084536,
    -0.98912701781200729,
    -0.99882619974750686,
    -1.0096013661471273,
    -1.0211190535988075,
    -1.0329791157406529,
    -1.0447220725757245,
    -1.0558388755542307,
    -1.0657828737906789,
    -1.0739836788967252,
    -1.0798625460670386,
    -1.0828488205838522,
    -1.0823969447776371,
    -1.0780034831810819,
    -1.0692236050257142,
    -1.0556864645490951,
    -1.037108941245712,
    -1.0133072438595792,
    -0.98420594244595727,
    -0.94984407033163909,
    -0.91037802969271242,
    -0.86608113756005511,
    -0.81733975968916606,
    -0.76464609388455973,
    -0.70858777785718585,
    -0.64983460530389081,
    -0.58912273356640099,
    -0.52723685319998248,
    -0.46499086076700791,
    -0.40320762847143365,
    -0.3426984958744288,
    -0.28424311867241758,
    -0.22857029700462772,
    -0.17634037146572068,
    -0.12812972023847746,
    -0.084417817617136071,
    -0.045577225441480018,
    -0.011866787962380831,
    0.016571808787668729,
    0.039714065848477365,
    0.057649355079744113,
    0.070574434369641503,
    0.078784072754749301,
    0.082659154155680925,
    0.082652703941148431,
    0.079274340792956144,
    0.073073695555439633,
    0.064623357877406695,
    0.054501910244962426,
    0.043277587996503941,
    0.031493064391093339,
    0.019651803733228279,
    0.0082063554927680471,
    -0.0024511186779023753,
    -0.011995881773299244,
    -0.020175111550195024,
    -0.02680982650460173,
    -0.031794320351297256,
    -0.035093261363296491,
    -0.036736687377621702,
    -0.036813190000957841,
    -0.035461630648150444,
    -0.032861765218153205,
    -0.029224172793483327,
    -0.024779886719031974,
    -0.01977011437063092,
    -0.014436406029201931,
    -0.0090115951983427145,
    -0.0037117845186128706,
    0.0012704044737910092,
    0.0057711605396715271,
    0.0096592971919957638,
    0.012838522078005843,
    0.015248191648030707,
    0.016862637248509005,
    0.017689197695827774,
    0.0177651338907856,
    0.017153631993356067,
    0.015939122421657403,
    0.014222152254877107,
    0.012114048738722571,
    0.0097316021553703891,
    0.0071919783297369477,
    0.0046080457993520341,
    0.0020842717001636757,
    -0.0002866946145224647,
    -0.0024266685640240704,
    -0.0042737642417056687,
    -0.0057834084561195764,
    -0.0069285625479086344,
    -0.0076992102328466983,
    -0.0081011955171294003,
    -0.008154514962190218,
    -0.0078911827906346099,
    -0.0073527953642053001,
    -0.0065879235522982027,
    -0.0056494578252792582,
    -0.004592022152574704,
    -0.0034695597424521107,
    -0.0023331772413973662,
    -0.0012293152097746507,
    -0.00019829253085423826,
    0.00072674810253947619,
    0.0015204864034918941,
    0.002165673682778646,
    0.002653012647775288,
    0.0029807091043616563,
    0.003153747445827193,
    0.0031829491114065966,
    0.0030838771627414996,
    0.002875650873139568,
    0.0025797320052745043,
    0.00221873963715713,
    0.001815343442062245,
    0.0013912767536239379,
    0.00096650110084373075,
    0.0005585437290764883,
    0.00018201945719154011,
    -0.00015166146478928777,
    -0.00043440662849036915,
    -0.00066139057558678438,
    -0.00083086055916281598,
    -0.00094382427706785066,
    -0.0010036477307536762,
    -0.0010155925592896242,
    -0.00098632186188263304,
    -0.00092340182309103014,
    -0.00083482361273923278,
    -0.00072856630138462553,
    -0.00061221718471593523,
    -0.00049266122252766831,
    -0.00037584653537969803,
    -0.00026662830753054756,
    -0.00016868922861601465,
    -8.4530939540528038e-05,
    -1.5527956662042541e-05,
    3.7966686986079523e-05,
    7.6476289097460182e-05,
    0.00010122868102335758,
    0.00011397057719086945,
    0.0001167801267734081,
    0.00011188843209297694,
    0.00010151907149702488,
    8.7752633005200309e-05,
    7.2421062316418983e-05,
    5.7034384291165807e-05,
    4.274019062067147e-05,
    3.0314303317569014e-05,
    2.0179311211787621e-05,
    1.2446301481450206e-05,
    6.9741154016256246e-06,
    3.4398698811610728e-06,
    1.4143057780469093e-06,
    4.3573249208783693e-07,
    7.6900572798521694e-08,
    0,
};

static const double g_sim_miniblamp_table[257] = {
    0,
    -0.0039062500129763891,
    -0.0078125002299248916,
    -0.011718751449476254,
    -0.01562500559936545,
    -0.019531266143148748,
    -0.023437538295037291,
    -0.027343828991080558,
    -0.031250146582060143,
    -0.035156500234316615,
    -0.039062899048234255,
    -0.042969350928942711,
    -0.046875861268492962,
    -0.050782431521765155,
    -0.054689057778082403,
    -0.058595729445411422,
    -0.062502428172745372,
    -0.066409127137631613,
    -0.070315790818984591,
    -0.074222375359852197,
    -0.078128829600670885,
    -0.082035096831229906,
    -0.085941117270062428,
    -0.089846831234800251,
    -0.093752182918164048,
    -0.097657124634135728,
    -0.10156162135023759,
    -0.10546565527771574,
    -0.10936923025486826,
    -0.11327237563277651,
    -0.11717514936004436,
    -0.12107763996615596,
    -0.12497996716345282,
    -0.12888228082647168,
    -0.1327847581645368,
    -0.13668759897812421,
    -0.1405910189796101,
    -0.14449524126150551,
    -0.14840048610607662,
    -0.15230695944435324,
    -0.156214840384201,
    -0.16012426833013643,
    -0.1640353303054275,
    -0.16794804915337702,
    -0.17186237333360557,
    -0.17577816903551449,
    -0.179695215300957,
    -0.18361320277900295,
    -0.18753173662683903,
    -0.19145034392357474,
    -0.19536848578140448,
    -0.19928557412673378,
    -0.20320099289010327,
    -0.20711412309750607,
    -0.2110243711080334,
    -0.21493119900589677,
    -0.21883415594161187,
    -0.22273290904039583,
    -0.22662727236795199,
    -0.23051723237586688,
    -0.2344029682499825,
    -0.23828486566194856,
    -0.24216352258023666,
    -0.24603974603220752,
    -0.2499145390195332,
    -0.25378907716758237,
    -0.25766467512355273,
    -0.26154274319276843,
    -0.26542473519899917,
    -0.26931208905163517,
    -0.27320616197699349,
    -0.27710816279900574,
    -0.28101908401232045,
    -0.28493963665605099,
    -0.28887019114911133,
    -0.29281072727198754,
    -0.29676079636319824,
    -0.30071949853538932,
    -0.30468547730593915,
    -0.30865693348667622,
    -0.31263165950014082,
    -0.31660709450569113,
    -0.32058039985375475,
    -0.32454855347215905,
    -0.32850846086058388,
    -0.3324570794666446,
    -0.33639155238031621,
    -0.34030934655255202,
    -0.34420839015728238,
    -0.34808720330801363,
    -0.35194501614004853,
    -0.35578186829895919,
    -0.35959868414905111,
    -0.36339731853645041,
    -0.36718056870432858,
    -0.37095214894649153,
    -0.37471662577373671,
    -0.37847931271904867,
    -0.38224612537832853,
    -0.38602339882129533,
    -0.389817671055529,
    -0.39363543772526088,
    -0.39748288461454379,
    -0.40136560574258973,
    -0.40528831583222769,
    -0.40925456665204396,
    -0.41326647713903508,
    -0.41732448727246552,
    -0.42142714537428183,
    -0.42557093785347111,
    -0.4297501694016887,
    -0.43395690030982104,
    -0.43818094594781104,
    -0.44240994158328267,
    -0.44662947366913952,
    -0.45082327657579341,
    -0.45497349155543171,
    -0.45906098258237471,
    -0.46306570169390848,
    -0.46696709463591146,
    -0.47074453606711769,
    -0.47437778235622774,
    -0.4778474291672683,
    -0.48113536060720818,
    -0.48422517672746934,
    -0.48710258663321493,
    -0.48975575535032639,
    -0.49217559390280741,
    -0.49435598372071049,
    -0.49629392847455228,
    -0.49798962864884611,
    -0.4994464765479591,
    -0.50067097188887089,
    -0.50167256059136511,
    -0.50246340174072124,
    -0.50305806988858104,
    -0.50347320179845534,
    -0.50372709836692919,
    -0.50383929370560865,
    -0.50383010421180929,
    -0.50372017086291054,
    -0.50353000793141012,
    -0.50327957084264174,
    -0.50298785500841436,
    -0.50267253620585495,
    -0.50234966148300952,
    -0.50203339772376321,
    -0.5017358429652703,
    -0.50146690340778421,
    -0.50123423686848267,
    -0.50104326128597976,
    -0.50089722485553523,
    -0.50079733253497993,
    -0.50074292206774162,
    -0.50073168137083757,
    -0.50075989816859379,
    -0.50082273213992878,
    -0.50091449959706702,
    -0.50102896082139492,
    -0.50115960062943121,
    -0.50129989349806581,
    -0.50144354560232085,
    -0.50158470736140115,
    -0.50171815149395249,
    -0.50183941309163149,
    -0.50194488977036689,
    -0.5020319014912451,
    -0.50209871110140103,
    -0.50214450797879862,
    -0.50216935832980825,
    -0.50217412665020833,
    -0.50216037359354138,
    -0.50213023598078421,
    -0.50208629492752244,
    -0.50203143806477624,
    -0.50196872160208772,
    -0.50190123754946203,
    -0.50183199080808194,
    -0.5017637900934645,
    -0.50169915580749769,
    -0.50164024706789512,
    -0.50158880917532944,
    -0.50154614188842694,
    -0.50151308802029193,
    -0.50149004109816475,
    -0.50147697016554849,
    -0.50147345927280307,
    -0.50147875881026116,
    -0.50149184559308491,
    -0.50151148850851035,
    -0.50153631657687758,
    -0.50156488644558994,
    -0.50159574661307038,
    -0.50162749604760026,
    -0.50165883530102373,
    -0.50168860869585741,
    -0.5017158366624912,
    -0.5017397377979943,
    -0.50175974068857609,
    -0.50177548596571486,
    -0.50178681943638648,
    -0.50179377742945519,
    -0.50179656572582365,
    -0.50179553358603524,
    -0.50179114445614059,
    -0.50178394492472211,
    -0.50177453342798273,
    -0.50176353006518559,
    -0.50175154870473604,
    -0.50173917234427268,
    -0.50172693244920596,
    -0.5017152927460109,
    -0.50170463770132645,
    -0.50169526568639977,
    -0.5016873866178857,
    -0.50168112368781603,
    -0.50167651865294405,
    -0.50167354004976061,
    -0.50167209363728749,
    -0.50167203434433361,
    -0.50167317900857833,
    -0.50167531923749253,
    -0.50167823379049015,
    -0.50168169997181089,
    -0.50168550362807618,
    -0.50168944745676769,
    -0.50169335744587151,
    -0.50169708737494367,
    -0.50170052140899801,
    -0.50170357490492401,
    -0.50170619362267033,
    -0.50170835158830951,
    -0.50171004789252416,
    -0.50171130272620168,
    -0.50171215295576443,
    -0.50171264752640532,
    -0.50171284295393692,
    -0.50171279912829181,
    -0.50171257560685412,
    -0.50171222852683439,
    -0.50171180821578321,
    -0.50171135753081453,
    -0.5017109109125355,
    -0.5017104941010051,
    -0.50171012442970719,
    -0.50170981159045847,
    -0.50170955874778933,
    -0.50170936387557274,
    -0.50170922119101424,
    -0.50170912257067335,
    -0.50170905884877359,
    -0.50170902091827185,
    -0.5017090005784568,
    -0.501708991097645,
    -0.50170898748428905,
    -0.50170898648305262,
    -0.50170898633285621,
};

/**
 * @brief Linearly interpolate the precomputed miniBLEP correction table.
 *
 * @param u Normalized table coordinate in [0, 1].
 * @return Interpolated miniBLEP correction value.
 */
static inline double sim_fourier_miniblep_eval(double u) {
    if (u <= 0.0) {
        return g_sim_miniblep_table[0];
    }
    if (u >= 1.0) {
        return g_sim_miniblep_table[SIM_FOURIER_MINIBLEP_TABLE_SIZE - 1];
    }

    double position = u * (double) (SIM_FOURIER_MINIBLEP_TABLE_SIZE - 1);
    size_t i        = (size_t) position;
    double frac     = position - (double) i;
    double a        = g_sim_miniblep_table[i];
    double b        = g_sim_miniblep_table[i + 1];
    return a + (b - a) * frac;
}

/**
 * @brief Linearly interpolate the precomputed miniBLAMP correction table.
 *
 * @param u Normalized table coordinate in [0, 1].
 * @return Interpolated miniBLAMP correction value.
 */
static inline double sim_fourier_miniblamp_eval(double u) {
    if (u <= 0.0) {
        return g_sim_miniblamp_table[0];
    }
    if (u >= 1.0) {
        return g_sim_miniblamp_table[SIM_FOURIER_MINIBLEP_TABLE_SIZE - 1];
    }

    double position = u * (double) (SIM_FOURIER_MINIBLEP_TABLE_SIZE - 1);
    size_t i        = (size_t) position;
    double frac     = position - (double) i;
    double a        = g_sim_miniblamp_table[i];
    double b        = g_sim_miniblamp_table[i + 1];
    return a + (b - a) * frac;
}

/**
 * @brief Compute the normalized miniBLEP support window for a phase increment.
 *
 * @param dphase Normalized phase increment in cycles per sample.
 * @return Window width, capped by `SIM_FOURIER_MINIBLEP_MAX_WINDOW`.
 */
static inline double sim_fourier_miniblep_window(double dphase) {
    double w = SIM_FOURIER_MINIBLEP_SPAN * dphase;
    if (w > SIM_FOURIER_MINIBLEP_MAX_WINDOW) {
        w = SIM_FOURIER_MINIBLEP_MAX_WINDOW;
    }
    return w;
}

/*------------------------------------------------------------------------------
 * BLIT kernels
 *------------------------------------------------------------------------------*/
/**
 * @brief Evaluate the normalized real Dirichlet kernel for a harmonic count.
 *
 * @param phase_radians Phase in radians.
 * @param harmonic_count Number of positive harmonics; negative returns zero.
 * @return Normalized Dirichlet sample with DC gain one.
 */
double sim_fourier_dirichlet(double phase_radians, int harmonic_count) {
    if (harmonic_count < 0) {
        return 0.0;
    }

    phase_radians = sim_fourier_wrap_two_pi(phase_radians);
    int    M_int  = sim_fourier_dirichlet_length(harmonic_count);
    double M      = (double) M_int;
    double half   = 0.5 * phase_radians;
    double ratio  = sim_fourier_sinc(M * half) / sim_fourier_sinc(half);
    return ratio;
}

/**
 * @brief Evaluate the complex Dirichlet kernel for a harmonic count.
 *
 * @param phase_radians Complex phase in radians.
 * @param harmonic_count Number of positive harmonics; negative returns zero.
 * @return Complex Dirichlet sample.
 */
SimComplexDouble sim_fourier_dirichlet_complex(SimComplexDouble phase_radians, int harmonic_count) {
    SimComplexDouble out = { 0.0, 0.0 };
    if (harmonic_count < 0) {
        return out;
    }

    phase_radians.re     = sim_fourier_wrap_two_pi(phase_radians.re);
    int            M_int = sim_fourier_dirichlet_length(harmonic_count);
    double complex z     = phase_radians.re + I * phase_radians.im;
    double complex half  = 0.5 * z;
    double complex num   = csin((double) M_int * half);
    double complex denom = csin(half);

    double       denom_mag2 = creal(denom) * creal(denom) + cimag(denom) * cimag(denom);
    const double eps2       = 1e-28;
    if (denom_mag2 < eps2) {
        out.re = 1.0;
        out.im = 0.0;
        return out;
    }

    double complex value = num / denom;
    out.re               = creal(value);
    out.im               = cimag(value);
    return out;
}

/**
 * @brief Evaluate the DC-corrected real BLIT kernel.
 *
 * @param phase_radians Phase in radians.
 * @param harmonic_count Number of positive harmonics.
 * @return Dirichlet sample minus `1 / (2N + 1)`.
 */
double sim_fourier_blit(double phase_radians, int harmonic_count) {
    if (harmonic_count < 0) {
        return 0.0;
    }
    int    M_int = sim_fourier_dirichlet_length(harmonic_count);
    double M     = (double) M_int;
    return sim_fourier_dirichlet(phase_radians, harmonic_count) - 1.0 / M;
}

/**
 * @brief Evaluate the DC-corrected complex BLIT kernel.
 *
 * @param phase_radians Complex phase in radians.
 * @param harmonic_count Number of positive harmonics.
 * @return Complex BLIT sample.
 */
SimComplexDouble sim_fourier_blit_complex(SimComplexDouble phase_radians, int harmonic_count) {
    SimComplexDouble out = sim_fourier_dirichlet_complex(phase_radians, harmonic_count);
    if (harmonic_count < 0) {
        return out;
    }
    int    M_int = sim_fourier_dirichlet_length(harmonic_count);
    double M     = (double) M_int;
    out.re -= 1.0 / M;
    return out;
}

/**
 * @brief Integrate BLIT samples into a bandlimited saw accumulator.
 *
 * @param phase_radians Phase in radians.
 * @param phase_increment_radians Per-sample phase increment, currently unused.
 * @param harmonic_count Number of positive harmonics.
 * @param state In/out integrator state.
 * @return Updated saw accumulator, or zero for invalid state/count.
 */
double sim_fourier_saw_blit(double  phase_radians,
                            double  phase_increment_radians,
                            int     harmonic_count,
                            double* state) {
    (void) phase_increment_radians;
    if (state == NULL || harmonic_count <= 0) {
        return 0.0;
    }

    double blit = sim_fourier_blit(phase_radians, harmonic_count);
    double y    = *state + blit;
    *state      = y;
    return y;
}

/**
 * @brief Integrate complex BLIT samples into a complex saw accumulator.
 *
 * @param phase_radians Complex phase in radians.
 * @param phase_increment_radians Per-sample phase increment, currently unused.
 * @param harmonic_count Number of positive harmonics.
 * @param state In/out complex integrator state.
 * @return Updated complex saw accumulator.
 */
SimComplexDouble sim_fourier_saw_blit_complex(SimComplexDouble  phase_radians,
                                              double            phase_increment_radians,
                                              int               harmonic_count,
                                              SimComplexDouble* state) {
    (void) phase_increment_radians;
    SimComplexDouble zero = { 0.0, 0.0 };
    if (state == NULL || harmonic_count <= 0) {
        return zero;
    }

    SimComplexDouble blit = sim_fourier_blit_complex(phase_radians, harmonic_count);
    SimComplexDouble y    = { state->re + blit.re, state->im + blit.im };
    *state                = y;
    return y;
}

/**
 * @brief Integrate a phase-shifted BLIT difference into a square waveform.
 *
 * @param phase_radians Phase in radians.
 * @param phase_increment_radians Per-sample phase increment, currently unused.
 * @param harmonic_count Number of positive harmonics.
 * @param duty Duty cycle, clamped to [0, 1] unless validation is assumed.
 * @param state In/out integrator state.
 * @return Updated square accumulator.
 */
double sim_fourier_square_blit(double  phase_radians,
                               double  phase_increment_radians,
                               int     harmonic_count,
                               double  duty,
                               double* state) {
    (void) phase_increment_radians;
    if (state == NULL || harmonic_count <= 0) {
        return 0.0;
    }

    double duty_wrapped  = sim_fourier_clamp01(duty);
    double shifted_phase = phase_radians + duty_wrapped * SIM_FOURIER_TWO_PI;
    double blit          = sim_fourier_blit(phase_radians, harmonic_count) -
                  sim_fourier_blit(shifted_phase, harmonic_count);

    double y = *state + blit;
    *state   = y;
    return y;
}

/**
 * @brief Integrate a complex phase-shifted BLIT difference into a square waveform.
 *
 * @param phase_radians Complex phase in radians.
 * @param phase_increment_radians Per-sample phase increment, currently unused.
 * @param harmonic_count Number of positive harmonics.
 * @param duty Duty cycle.
 * @param state In/out complex integrator state.
 * @return Updated complex square accumulator.
 */
SimComplexDouble sim_fourier_square_blit_complex(SimComplexDouble  phase_radians,
                                                 double            phase_increment_radians,
                                                 int               harmonic_count,
                                                 double            duty,
                                                 SimComplexDouble* state) {
    (void) phase_increment_radians;
    SimComplexDouble zero = { 0.0, 0.0 };
    if (state == NULL || harmonic_count <= 0) {
        return zero;
    }

    double           duty_wrapped  = sim_fourier_clamp01(duty);
    double           shifted_phase = phase_radians.re + duty_wrapped * SIM_FOURIER_TWO_PI;
    SimComplexDouble blit_a        = sim_fourier_blit_complex(phase_radians, harmonic_count);
    SimComplexDouble blit_b        = sim_fourier_blit_complex(
        (SimComplexDouble) { shifted_phase, phase_radians.im }, harmonic_count);
    SimComplexDouble diff = { blit_a.re - blit_b.re, blit_a.im - blit_b.im };
    SimComplexDouble y    = { state->re + diff.re, state->im + diff.im };
    *state                = y;
    return y;
}

/**
 * @brief Double-integrate a BLIT square into a triangle waveform.
 *
 * @param phase_radians Phase in radians.
 * @param phase_increment_radians Per-sample phase increment.
 * @param harmonic_count Number of positive harmonics.
 * @param velocity_state First integrator state.
 * @param position_state Second integrator state.
 * @return Wrapped bipolar triangle value.
 */
double sim_fourier_triangle_blit(double  phase_radians,
                                 double  phase_increment_radians,
                                 int     harmonic_count,
                                 double* velocity_state,
                                 double* position_state) {
    (void) phase_increment_radians;
    if (velocity_state == NULL || position_state == NULL || harmonic_count <= 0) {
        return 0.0;
    }

    double dphase_cycles = phase_increment_radians / SIM_FOURIER_TWO_PI;
    double slope         = 4.0 * dphase_cycles; /* gain for frequency-invariant amplitude */
    double square =
        sim_fourier_square_blit(phase_radians, 0.0, harmonic_count, 0.5, velocity_state);
    double tri      = *position_state + square * slope;
    tri             = sim_fourier_center_bipolar(tri);
    *position_state = tri;
    return tri;
}

/**
 * @brief Double-integrate a complex BLIT square into a complex triangle waveform.
 *
 * @param phase_radians Complex phase in radians.
 * @param phase_increment_radians Per-sample phase increment.
 * @param harmonic_count Number of positive harmonics.
 * @param velocity_state First complex integrator state.
 * @param position_state Second complex integrator state.
 * @return Wrapped complex triangle value.
 */
SimComplexDouble sim_fourier_triangle_blit_complex(SimComplexDouble  phase_radians,
                                                   double            phase_increment_radians,
                                                   int               harmonic_count,
                                                   SimComplexDouble* velocity_state,
                                                   SimComplexDouble* position_state) {
    (void) phase_increment_radians;
    SimComplexDouble zero = { 0.0, 0.0 };
    if (velocity_state == NULL || position_state == NULL || harmonic_count <= 0) {
        return zero;
    }

    double           dphase_cycles = phase_increment_radians / SIM_FOURIER_TWO_PI;
    double           slope         = 4.0 * dphase_cycles;
    SimComplexDouble square =
        sim_fourier_square_blit_complex(phase_radians, 0.0, harmonic_count, 0.5, velocity_state);
    SimComplexDouble tri = { position_state->re + square.re * slope,
                             position_state->im + square.im * slope };
    tri.re               = sim_fourier_center_bipolar(tri.re);
    tri.im               = sim_fourier_center_bipolar(tri.im);
    *position_state      = tri;
    return tri;
}

/*------------------------------------------------------------------------------
 * PolyBLEP / PolyBLAMP
 *------------------------------------------------------------------------------*/
/**
 * @brief Evaluate a two-sample PolyBLEP correction for a step discontinuity.
 *
 * @param phase Normalized phase in [0, 1).
 * @param dphase Normalized phase increment.
 * @return Correction value to subtract from a naive step waveform.
 */
double sim_fourier_polyblep(double phase, double dphase) {
    if (dphase <= 0.0) {
        return 0.0;
    }
    double p   = sim_fourier_clamp01(phase);
    double inv = 1.0 / dphase;

    if (p < dphase) {
        double t = p * inv;
        return t + t - t * t - 1.0;
    }
    if (p > 1.0 - dphase) {
        double t = (p - 1.0) * inv;
        return t * t + 2.0 * t + 1.0;
    }
    return 0.0;
}

/**
 * @brief Evaluate PolyBLEP as a real-valued complex result.
 *
 * @param phase Normalized phase in [0, 1).
 * @param dphase Normalized phase increment.
 * @return Complex value with zero imaginary component.
 */
SimComplexDouble sim_fourier_polyblep_complex(double phase, double dphase) {
    SimComplexDouble out = { sim_fourier_polyblep(phase, dphase), 0.0 };
    return out;
}

/**
 * @brief Evaluate the integrated PolyBLEP correction for slope discontinuities.
 *
 * @param phase Normalized phase in [0, 1).
 * @param dphase Normalized phase increment.
 * @return PolyBLAMP correction value.
 */
double sim_fourier_polyblamp(double phase, double dphase) {
    if (dphase <= 0.0) {
        return 0.0;
    }

    double p   = sim_fourier_clamp01(phase);
    double inv = 1.0 / dphase;

    if (p < dphase) {
        double x  = p * inv;
        double x2 = x * x;
        double x3 = x2 * x;
        return dphase * (x2 - x - (1.0 / 3.0) * x3);
    }
    if (p > 1.0 - dphase) {
        double x  = (p - 1.0) * inv;
        double x2 = x * x;
        double x3 = x2 * x;
        return (-dphase / 3.0) + dphase * ((1.0 / 3.0) * x3 + x2 + x + 1.0 / 3.0);
    }
    return -dphase / 3.0;
}

/**
 * @brief Evaluate PolyBLAMP as a real-valued complex result.
 *
 * @param phase Normalized phase in [0, 1).
 * @param dphase Normalized phase increment.
 * @return Complex value with zero imaginary component.
 */
SimComplexDouble sim_fourier_polyblamp_complex(double phase, double dphase) {
    SimComplexDouble out = { sim_fourier_polyblamp(phase, dphase), 0.0 };
    return out;
}

/*------------------------------------------------------------------------------
 * miniBLEP / miniBLAMP
 *------------------------------------------------------------------------------*/
/**
 * @brief Evaluate the table-driven miniBLEP correction.
 *
 * @param phase Normalized phase in [0, 1).
 * @param dphase Normalized phase increment.
 * @return Correction value to subtract from a naive discontinuity.
 */
double sim_fourier_miniblep(double phase, double dphase) {
    if (dphase <= 0.0) {
        return 0.0;
    }

    double p      = sim_fourier_clamp01(phase);
    double window = sim_fourier_miniblep_window(dphase);
    if (window <= 0.0) {
        return 0.0;
    }
    double inv_window = 1.0 / window;

    if (p < window) {
        return sim_fourier_miniblep_eval(p * inv_window);
    }

    double tail = 1.0 - p;
    if (tail < window) {
        return -sim_fourier_miniblep_eval(tail * inv_window);
    }

    return 0.0;
}

/**
 * @brief Evaluate miniBLEP as a real-valued complex result.
 *
 * @param phase Normalized phase in [0, 1).
 * @param dphase Normalized phase increment.
 * @return Complex value with zero imaginary component.
 */
SimComplexDouble sim_fourier_miniblep_complex(double phase, double dphase) {
    SimComplexDouble out = { sim_fourier_miniblep(phase, dphase), 0.0 };
    return out;
}

/**
 * @brief Evaluate the integrated table-driven miniBLEP correction.
 *
 * @param phase Normalized phase in [0, 1).
 * @param dphase Normalized phase increment.
 * @return miniBLAMP correction value.
 */
double sim_fourier_miniblamp(double phase, double dphase) {
    if (dphase <= 0.0) {
        return 0.0;
    }

    double p      = sim_fourier_clamp01(phase);
    double window = sim_fourier_miniblep_window(dphase);
    if (window <= 0.0) {
        return 0.0;
    }
    double inv_window = 1.0 / window;

    if (p < window) {
        return window * sim_fourier_miniblamp_eval(p * inv_window);
    }

    double tail = 1.0 - p;
    if (tail < window) {
        double leading = window * SIM_FOURIER_MINIBLEP_AREA;
        return leading - window * sim_fourier_miniblamp_eval(tail * inv_window);
    }

    return window * SIM_FOURIER_MINIBLEP_AREA;
}

/**
 * @brief Evaluate miniBLAMP as a real-valued complex result.
 *
 * @param phase Normalized phase in [0, 1).
 * @param dphase Normalized phase increment.
 * @return Complex value with zero imaginary component.
 */
SimComplexDouble sim_fourier_miniblamp_complex(double phase, double dphase) {
    SimComplexDouble out = { sim_fourier_miniblamp(phase, dphase), 0.0 };
    return out;
}
