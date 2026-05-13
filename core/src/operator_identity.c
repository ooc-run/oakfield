/**
 * @file operator_identity.c
 * @brief Stable string conversion for core operator identity enums.
 *
 * The identity helpers translate between public enum values and schema-facing
 * names for IR opcodes and warp levels. Parsing is case-insensitive for caller
 * convenience but returns explicit false/sentinel values for unknown text.
 */
#include "oakfield/operator_identity.h"

#include <ctype.h>
#include <stddef.h>
#include <string.h>

static bool sim_ascii_equal_ci(const char* lhs, const char* rhs) {
    if (lhs == NULL || rhs == NULL) {
        return false;
    }

    while (*lhs != '\0' && *rhs != '\0') {
        unsigned char a = (unsigned char) *lhs++;
        unsigned char b = (unsigned char) *rhs++;
        if (tolower(a) != tolower(b)) {
            return false;
        }
    }
    return *lhs == '\0' && *rhs == '\0';
}

static bool sim_ascii_starts_with_ci(const char* text, const char* prefix) {
    if (text == NULL || prefix == NULL) {
        return false;
    }

    while (*prefix != '\0') {
        unsigned char a = (unsigned char) *text++;
        unsigned char b = (unsigned char) *prefix++;
        if (tolower(a) != tolower(b)) {
            return false;
        }
    }
    return true;
}

const char* sim_ir_opcode_name(SimIROpcode opcode) {
    switch (opcode) {
        case OAK_OP_DIFF:
            return "diff";
        case OAK_OP_CONV:
            return "conv";
        case OAK_OP_DISP:
            return "disp";
        case OAK_OP_DIFFUSE:
            return "diffuse";
        case OAK_OP_WARP:
            return "warp";
        case OAK_OP_NOISE:
            return "noise";
        case OAK_OP_FLOW:
            return "flow";
        case OAK_OP_CORE:
            return "core";
        case OAK_OP_MISC:
        default:
            return "misc";
    }
}

bool sim_ir_opcode_from_string(const char* text, SimIROpcode* out_opcode) {
    const char* key = text;

    if (text == NULL || out_opcode == NULL) {
        return false;
    }

    if (sim_ascii_starts_with_ci(key, "oak_op_")) {
        key += strlen("oak_op_");
    }

    if (sim_ascii_equal_ci(key, "diff")) {
        *out_opcode = OAK_OP_DIFF;
        return true;
    }
    if (sim_ascii_equal_ci(key, "conv") || sim_ascii_equal_ci(key, "convolution")) {
        *out_opcode = OAK_OP_CONV;
        return true;
    }
    if (sim_ascii_equal_ci(key, "disp") || sim_ascii_equal_ci(key, "dispersion")) {
        *out_opcode = OAK_OP_DISP;
        return true;
    }
    if (sim_ascii_equal_ci(key, "diffuse") || sim_ascii_equal_ci(key, "diffusion")) {
        *out_opcode = OAK_OP_DIFFUSE;
        return true;
    }
    if (sim_ascii_equal_ci(key, "warp")) {
        *out_opcode = OAK_OP_WARP;
        return true;
    }
    if (sim_ascii_equal_ci(key, "noise")) {
        *out_opcode = OAK_OP_NOISE;
        return true;
    }
    if (sim_ascii_equal_ci(key, "flow")) {
        *out_opcode = OAK_OP_FLOW;
        return true;
    }
    if (sim_ascii_equal_ci(key, "core")) {
        *out_opcode = OAK_OP_CORE;
        return true;
    }
    if (sim_ascii_equal_ci(key, "misc")) {
        *out_opcode = OAK_OP_MISC;
        return true;
    }

    return false;
}
