//! Unicode display width utilities.
//!
//! Provides accurate terminal column width calculation for UTF-8 text,
//! including East Asian Width property support for CJK, emoji, and
//! combining marks.

const std = @import("std");

/// A Unicode codepoint range with associated display width.
const WidthRange = struct {
    first: u21,
    last: u21,
    width: u2,
};

/// Zero-width codepoint ranges: combining marks, zero-width joiners,
/// variation selectors, and other non-spacing characters.
const zero_width_ranges = [_]WidthRange{
    .{ .first = 0x0000, .last = 0x001F, .width = 0 }, // C0 control
    .{ .first = 0x007F, .last = 0x009F, .width = 0 }, // DEL + C1
    .{ .first = 0x00AD, .last = 0x00AD, .width = 0 }, // soft hyphen
    .{ .first = 0x0300, .last = 0x036F, .width = 0 }, // combining diacriticals
    .{ .first = 0x0483, .last = 0x0489, .width = 0 }, // cyrillic combining
    .{ .first = 0x0591, .last = 0x05BD, .width = 0 }, // hebrew combining
    .{ .first = 0x05BF, .last = 0x05BF, .width = 0 },
    .{ .first = 0x05C1, .last = 0x05C2, .width = 0 },
    .{ .first = 0x05C4, .last = 0x05C5, .width = 0 },
    .{ .first = 0x05C7, .last = 0x05C7, .width = 0 },
    .{ .first = 0x0600, .last = 0x0605, .width = 0 }, // arabic number sign
    .{ .first = 0x0610, .last = 0x061A, .width = 0 }, // arabic combining
    .{ .first = 0x061C, .last = 0x061C, .width = 0 }, // arabic letter mark
    .{ .first = 0x064B, .last = 0x065F, .width = 0 }, // arabic combining
    .{ .first = 0x0670, .last = 0x0670, .width = 0 },
    .{ .first = 0x06D6, .last = 0x06DD, .width = 0 },
    .{ .first = 0x06DF, .last = 0x06E4, .width = 0 },
    .{ .first = 0x06E7, .last = 0x06E8, .width = 0 },
    .{ .first = 0x06EA, .last = 0x06ED, .width = 0 },
    .{ .first = 0x070F, .last = 0x070F, .width = 0 }, // syriac abbreviation
    .{ .first = 0x0711, .last = 0x0711, .width = 0 },
    .{ .first = 0x0730, .last = 0x074A, .width = 0 }, // syriac combining
    .{ .first = 0x07A6, .last = 0x07B0, .width = 0 }, // thaana combining
    .{ .first = 0x07EB, .last = 0x07F3, .width = 0 }, // nko combining
    .{ .first = 0x07FD, .last = 0x07FD, .width = 0 },
    .{ .first = 0x0816, .last = 0x0819, .width = 0 }, // samaritan
    .{ .first = 0x081B, .last = 0x0823, .width = 0 },
    .{ .first = 0x0825, .last = 0x0827, .width = 0 },
    .{ .first = 0x0829, .last = 0x082D, .width = 0 },
    .{ .first = 0x0859, .last = 0x085B, .width = 0 },
    .{ .first = 0x0890, .last = 0x0891, .width = 0 },
    .{ .first = 0x0898, .last = 0x089F, .width = 0 },
    .{ .first = 0x08CA, .last = 0x08E1, .width = 0 }, // arabic combining
    .{ .first = 0x08E3, .last = 0x0902, .width = 0 },
    .{ .first = 0x093A, .last = 0x093A, .width = 0 }, // devanagari
    .{ .first = 0x093C, .last = 0x093C, .width = 0 },
    .{ .first = 0x0941, .last = 0x0948, .width = 0 },
    .{ .first = 0x094D, .last = 0x094D, .width = 0 },
    .{ .first = 0x0951, .last = 0x0957, .width = 0 },
    .{ .first = 0x0962, .last = 0x0963, .width = 0 },
    .{ .first = 0x0981, .last = 0x0981, .width = 0 }, // bengali
    .{ .first = 0x09BC, .last = 0x09BC, .width = 0 },
    .{ .first = 0x09C1, .last = 0x09C4, .width = 0 },
    .{ .first = 0x09CD, .last = 0x09CD, .width = 0 },
    .{ .first = 0x09E2, .last = 0x09E3, .width = 0 },
    .{ .first = 0x09FE, .last = 0x09FE, .width = 0 },
    .{ .first = 0x0A01, .last = 0x0A02, .width = 0 }, // gurmukhi
    .{ .first = 0x0A3C, .last = 0x0A3C, .width = 0 },
    .{ .first = 0x0A41, .last = 0x0A42, .width = 0 },
    .{ .first = 0x0A47, .last = 0x0A48, .width = 0 },
    .{ .first = 0x0A4B, .last = 0x0A4D, .width = 0 },
    .{ .first = 0x0A51, .last = 0x0A51, .width = 0 },
    .{ .first = 0x0A70, .last = 0x0A71, .width = 0 },
    .{ .first = 0x0A75, .last = 0x0A75, .width = 0 },
    .{ .first = 0x0A81, .last = 0x0A82, .width = 0 }, // gujarati
    .{ .first = 0x0ABC, .last = 0x0ABC, .width = 0 },
    .{ .first = 0x0AC1, .last = 0x0AC5, .width = 0 },
    .{ .first = 0x0AC7, .last = 0x0AC8, .width = 0 },
    .{ .first = 0x0ACD, .last = 0x0ACD, .width = 0 },
    .{ .first = 0x0AE2, .last = 0x0AE3, .width = 0 },
    .{ .first = 0x0AFA, .last = 0x0AFF, .width = 0 },
    .{ .first = 0x0B01, .last = 0x0B01, .width = 0 }, // oriya
    .{ .first = 0x0B3C, .last = 0x0B3C, .width = 0 },
    .{ .first = 0x0B3F, .last = 0x0B3F, .width = 0 },
    .{ .first = 0x0B41, .last = 0x0B44, .width = 0 },
    .{ .first = 0x0B4D, .last = 0x0B4D, .width = 0 },
    .{ .first = 0x0B55, .last = 0x0B56, .width = 0 },
    .{ .first = 0x0B62, .last = 0x0B63, .width = 0 },
    .{ .first = 0x0B82, .last = 0x0B82, .width = 0 }, // tamil
    .{ .first = 0x0BC0, .last = 0x0BC0, .width = 0 },
    .{ .first = 0x0BCD, .last = 0x0BCD, .width = 0 },
    .{ .first = 0x0C00, .last = 0x0C00, .width = 0 }, // telugu
    .{ .first = 0x0C04, .last = 0x0C04, .width = 0 },
    .{ .first = 0x0C3C, .last = 0x0C3C, .width = 0 },
    .{ .first = 0x0C3E, .last = 0x0C40, .width = 0 },
    .{ .first = 0x0C46, .last = 0x0C48, .width = 0 },
    .{ .first = 0x0C4A, .last = 0x0C4D, .width = 0 },
    .{ .first = 0x0C55, .last = 0x0C56, .width = 0 },
    .{ .first = 0x0C62, .last = 0x0C63, .width = 0 },
    .{ .first = 0x0C81, .last = 0x0C81, .width = 0 }, // kannada
    .{ .first = 0x0CBC, .last = 0x0CBC, .width = 0 },
    .{ .first = 0x0CBF, .last = 0x0CBF, .width = 0 },
    .{ .first = 0x0CC6, .last = 0x0CC6, .width = 0 },
    .{ .first = 0x0CCC, .last = 0x0CCD, .width = 0 },
    .{ .first = 0x0CE2, .last = 0x0CE3, .width = 0 },
    .{ .first = 0x0D00, .last = 0x0D01, .width = 0 }, // malayalam
    .{ .first = 0x0D3B, .last = 0x0D3C, .width = 0 },
    .{ .first = 0x0D41, .last = 0x0D44, .width = 0 },
    .{ .first = 0x0D4D, .last = 0x0D4D, .width = 0 },
    .{ .first = 0x0D62, .last = 0x0D63, .width = 0 },
    .{ .first = 0x0DCA, .last = 0x0DCA, .width = 0 }, // sinhala
    .{ .first = 0x0DD2, .last = 0x0DD4, .width = 0 },
    .{ .first = 0x0DD6, .last = 0x0DD6, .width = 0 },
    .{ .first = 0x0E31, .last = 0x0E31, .width = 0 }, // thai
    .{ .first = 0x0E34, .last = 0x0E3A, .width = 0 },
    .{ .first = 0x0E47, .last = 0x0E4E, .width = 0 },
    .{ .first = 0x0EB1, .last = 0x0EB1, .width = 0 }, // lao
    .{ .first = 0x0EB4, .last = 0x0EBC, .width = 0 },
    .{ .first = 0x0EC8, .last = 0x0ECE, .width = 0 },
    .{ .first = 0x0F18, .last = 0x0F19, .width = 0 }, // tibetan
    .{ .first = 0x0F35, .last = 0x0F35, .width = 0 },
    .{ .first = 0x0F37, .last = 0x0F37, .width = 0 },
    .{ .first = 0x0F39, .last = 0x0F39, .width = 0 },
    .{ .first = 0x0F71, .last = 0x0F7E, .width = 0 },
    .{ .first = 0x0F80, .last = 0x0F84, .width = 0 },
    .{ .first = 0x0F86, .last = 0x0F87, .width = 0 },
    .{ .first = 0x0F8D, .last = 0x0F97, .width = 0 },
    .{ .first = 0x0F99, .last = 0x0FBC, .width = 0 },
    .{ .first = 0x0FC6, .last = 0x0FC6, .width = 0 },
    .{ .first = 0x102D, .last = 0x1030, .width = 0 }, // myanmar
    .{ .first = 0x1032, .last = 0x1037, .width = 0 },
    .{ .first = 0x1039, .last = 0x103A, .width = 0 },
    .{ .first = 0x103D, .last = 0x103E, .width = 0 },
    .{ .first = 0x1058, .last = 0x1059, .width = 0 },
    .{ .first = 0x105E, .last = 0x1060, .width = 0 },
    .{ .first = 0x1071, .last = 0x1074, .width = 0 },
    .{ .first = 0x1082, .last = 0x1082, .width = 0 },
    .{ .first = 0x1085, .last = 0x1086, .width = 0 },
    .{ .first = 0x108D, .last = 0x108D, .width = 0 },
    .{ .first = 0x109D, .last = 0x109D, .width = 0 },
    .{ .first = 0x1160, .last = 0x11FF, .width = 0 }, // hangul jungseong/jongseong
    .{ .first = 0x135D, .last = 0x135F, .width = 0 }, // ethiopic combining
    .{ .first = 0x1712, .last = 0x1714, .width = 0 }, // tagalog
    .{ .first = 0x1732, .last = 0x1734, .width = 0 }, // hanunoo
    .{ .first = 0x1752, .last = 0x1753, .width = 0 }, // buhid
    .{ .first = 0x1772, .last = 0x1773, .width = 0 }, // tagbanwa
    .{ .first = 0x17B4, .last = 0x17B5, .width = 0 }, // khmer
    .{ .first = 0x17B7, .last = 0x17BD, .width = 0 },
    .{ .first = 0x17C6, .last = 0x17C6, .width = 0 },
    .{ .first = 0x17C9, .last = 0x17D3, .width = 0 },
    .{ .first = 0x17DD, .last = 0x17DD, .width = 0 },
    .{ .first = 0x180B, .last = 0x180E, .width = 0 }, // mongolian
    .{ .first = 0x1885, .last = 0x1886, .width = 0 },
    .{ .first = 0x18A9, .last = 0x18A9, .width = 0 },
    .{ .first = 0x1920, .last = 0x1922, .width = 0 }, // buginese
    .{ .first = 0x1927, .last = 0x1928, .width = 0 },
    .{ .first = 0x1932, .last = 0x1932, .width = 0 },
    .{ .first = 0x1939, .last = 0x193B, .width = 0 },
    .{ .first = 0x1A17, .last = 0x1A18, .width = 0 },
    .{ .first = 0x1A1B, .last = 0x1A1B, .width = 0 },
    .{ .first = 0x1A56, .last = 0x1A56, .width = 0 },
    .{ .first = 0x1A58, .last = 0x1A5E, .width = 0 },
    .{ .first = 0x1A60, .last = 0x1A60, .width = 0 },
    .{ .first = 0x1A62, .last = 0x1A62, .width = 0 },
    .{ .first = 0x1A65, .last = 0x1A6C, .width = 0 },
    .{ .first = 0x1A73, .last = 0x1A7C, .width = 0 },
    .{ .first = 0x1A7F, .last = 0x1A7F, .width = 0 },
    .{ .first = 0x1AB0, .last = 0x1ACE, .width = 0 }, // combining diacriticals ext
    .{ .first = 0x1B00, .last = 0x1B03, .width = 0 }, // balinese
    .{ .first = 0x1B34, .last = 0x1B34, .width = 0 },
    .{ .first = 0x1B36, .last = 0x1B3A, .width = 0 },
    .{ .first = 0x1B3C, .last = 0x1B3C, .width = 0 },
    .{ .first = 0x1B42, .last = 0x1B42, .width = 0 },
    .{ .first = 0x1B6B, .last = 0x1B73, .width = 0 },
    .{ .first = 0x1B80, .last = 0x1B81, .width = 0 },
    .{ .first = 0x1BA2, .last = 0x1BA5, .width = 0 },
    .{ .first = 0x1BA8, .last = 0x1BA9, .width = 0 },
    .{ .first = 0x1BAB, .last = 0x1BAD, .width = 0 },
    .{ .first = 0x1BE6, .last = 0x1BE6, .width = 0 },
    .{ .first = 0x1BE8, .last = 0x1BE9, .width = 0 },
    .{ .first = 0x1BED, .last = 0x1BED, .width = 0 },
    .{ .first = 0x1BEF, .last = 0x1BF1, .width = 0 },
    .{ .first = 0x1C2C, .last = 0x1C33, .width = 0 },
    .{ .first = 0x1C36, .last = 0x1C37, .width = 0 },
    .{ .first = 0x1CD0, .last = 0x1CD2, .width = 0 }, // vedic extensions
    .{ .first = 0x1CD4, .last = 0x1CE0, .width = 0 },
    .{ .first = 0x1CE2, .last = 0x1CE8, .width = 0 },
    .{ .first = 0x1CED, .last = 0x1CED, .width = 0 },
    .{ .first = 0x1CF4, .last = 0x1CF4, .width = 0 },
    .{ .first = 0x1CF8, .last = 0x1CF9, .width = 0 },
    .{ .first = 0x1DC0, .last = 0x1DFF, .width = 0 }, // combining diacriticals supp
    .{ .first = 0x200B, .last = 0x200F, .width = 0 }, // zero-width + direction
    .{ .first = 0x2028, .last = 0x202E, .width = 0 }, // line/paragraph sep + dir
    .{ .first = 0x2060, .last = 0x2064, .width = 0 }, // word joiner etc
    .{ .first = 0x2066, .last = 0x206F, .width = 0 }, // bidi isolates
    .{ .first = 0x20D0, .last = 0x20F0, .width = 0 }, // combining for symbols
    .{ .first = 0xFE00, .last = 0xFE0F, .width = 0 }, // variation selectors
    .{ .first = 0xFE20, .last = 0xFE2F, .width = 0 }, // combining half marks
    .{ .first = 0xFEFF, .last = 0xFEFF, .width = 0 }, // BOM / ZWNBSP
    .{ .first = 0xFFF9, .last = 0xFFFB, .width = 0 }, // interlinear annotation
    .{ .first = 0x101FD, .last = 0x101FD, .width = 0 }, // phaistos combining
    .{ .first = 0x102E0, .last = 0x102E0, .width = 0 },
    .{ .first = 0x10376, .last = 0x1037A, .width = 0 },
    .{ .first = 0x10A01, .last = 0x10A03, .width = 0 },
    .{ .first = 0x10A05, .last = 0x10A06, .width = 0 },
    .{ .first = 0x10A0C, .last = 0x10A0F, .width = 0 },
    .{ .first = 0x10A38, .last = 0x10A3A, .width = 0 },
    .{ .first = 0x10A3F, .last = 0x10A3F, .width = 0 },
    .{ .first = 0x10AE5, .last = 0x10AE6, .width = 0 },
    .{ .first = 0x10D24, .last = 0x10D27, .width = 0 },
    .{ .first = 0x10EAB, .last = 0x10EAC, .width = 0 },
    .{ .first = 0x10F46, .last = 0x10F50, .width = 0 },
    .{ .first = 0x11001, .last = 0x11001, .width = 0 },
    .{ .first = 0x11038, .last = 0x11046, .width = 0 },
    .{ .first = 0x11070, .last = 0x11070, .width = 0 },
    .{ .first = 0x11073, .last = 0x11074, .width = 0 },
    .{ .first = 0x1107F, .last = 0x11081, .width = 0 },
    .{ .first = 0x110B3, .last = 0x110B6, .width = 0 },
    .{ .first = 0x110B9, .last = 0x110BA, .width = 0 },
    .{ .first = 0x110C2, .last = 0x110C2, .width = 0 },
    .{ .first = 0x11100, .last = 0x11102, .width = 0 },
    .{ .first = 0x11127, .last = 0x1112B, .width = 0 },
    .{ .first = 0x1112D, .last = 0x11134, .width = 0 },
    .{ .first = 0x11173, .last = 0x11173, .width = 0 },
    .{ .first = 0x11180, .last = 0x11181, .width = 0 },
    .{ .first = 0x111B6, .last = 0x111BE, .width = 0 },
    .{ .first = 0x111C9, .last = 0x111CC, .width = 0 },
    .{ .first = 0x111CF, .last = 0x111CF, .width = 0 },
    .{ .first = 0x1122F, .last = 0x11231, .width = 0 },
    .{ .first = 0x11234, .last = 0x11234, .width = 0 },
    .{ .first = 0x11236, .last = 0x11237, .width = 0 },
    .{ .first = 0x1123E, .last = 0x1123E, .width = 0 },
    .{ .first = 0x112DF, .last = 0x112DF, .width = 0 },
    .{ .first = 0x112E3, .last = 0x112EA, .width = 0 },
    .{ .first = 0x11300, .last = 0x11301, .width = 0 },
    .{ .first = 0x1133B, .last = 0x1133C, .width = 0 },
    .{ .first = 0x11340, .last = 0x11340, .width = 0 },
    .{ .first = 0x11366, .last = 0x1136C, .width = 0 },
    .{ .first = 0x11370, .last = 0x11374, .width = 0 },
    .{ .first = 0x11438, .last = 0x1143F, .width = 0 },
    .{ .first = 0x11442, .last = 0x11444, .width = 0 },
    .{ .first = 0x11446, .last = 0x11446, .width = 0 },
    .{ .first = 0x1145E, .last = 0x1145E, .width = 0 },
    .{ .first = 0x114B3, .last = 0x114B8, .width = 0 },
    .{ .first = 0x114BA, .last = 0x114BA, .width = 0 },
    .{ .first = 0x114BF, .last = 0x114C0, .width = 0 },
    .{ .first = 0x114C2, .last = 0x114C3, .width = 0 },
    .{ .first = 0x115B2, .last = 0x115B5, .width = 0 },
    .{ .first = 0x115BC, .last = 0x115BD, .width = 0 },
    .{ .first = 0x115BF, .last = 0x115C0, .width = 0 },
    .{ .first = 0x115DC, .last = 0x115DD, .width = 0 },
    .{ .first = 0x11633, .last = 0x1163A, .width = 0 },
    .{ .first = 0x1163D, .last = 0x1163D, .width = 0 },
    .{ .first = 0x1163F, .last = 0x11640, .width = 0 },
    .{ .first = 0x116AB, .last = 0x116AB, .width = 0 },
    .{ .first = 0x116AD, .last = 0x116AD, .width = 0 },
    .{ .first = 0x116B0, .last = 0x116B5, .width = 0 },
    .{ .first = 0x116B7, .last = 0x116B7, .width = 0 },
    .{ .first = 0x1171D, .last = 0x1171F, .width = 0 },
    .{ .first = 0x11722, .last = 0x11725, .width = 0 },
    .{ .first = 0x11727, .last = 0x1172B, .width = 0 },
    .{ .first = 0x1182F, .last = 0x11837, .width = 0 },
    .{ .first = 0x11839, .last = 0x1183A, .width = 0 },
    .{ .first = 0x1193B, .last = 0x1193C, .width = 0 },
    .{ .first = 0x1193E, .last = 0x1193E, .width = 0 },
    .{ .first = 0x11943, .last = 0x11943, .width = 0 },
    .{ .first = 0x119D4, .last = 0x119D7, .width = 0 },
    .{ .first = 0x119DA, .last = 0x119DB, .width = 0 },
    .{ .first = 0x119E0, .last = 0x119E0, .width = 0 },
    .{ .first = 0x11A01, .last = 0x11A0A, .width = 0 },
    .{ .first = 0x11A33, .last = 0x11A38, .width = 0 },
    .{ .first = 0x11A3B, .last = 0x11A3E, .width = 0 },
    .{ .first = 0x11A47, .last = 0x11A47, .width = 0 },
    .{ .first = 0x11A51, .last = 0x11A56, .width = 0 },
    .{ .first = 0x11A59, .last = 0x11A5B, .width = 0 },
    .{ .first = 0x11A8A, .last = 0x11A96, .width = 0 },
    .{ .first = 0x11A98, .last = 0x11A99, .width = 0 },
    .{ .first = 0x11C30, .last = 0x11C36, .width = 0 },
    .{ .first = 0x11C38, .last = 0x11C3D, .width = 0 },
    .{ .first = 0x11C3F, .last = 0x11C3F, .width = 0 },
    .{ .first = 0x11C92, .last = 0x11CA7, .width = 0 },
    .{ .first = 0x11CAA, .last = 0x11CB0, .width = 0 },
    .{ .first = 0x11CB2, .last = 0x11CB3, .width = 0 },
    .{ .first = 0x11CB5, .last = 0x11CB6, .width = 0 },
    .{ .first = 0x11D31, .last = 0x11D36, .width = 0 },
    .{ .first = 0x11D3A, .last = 0x11D3A, .width = 0 },
    .{ .first = 0x11D3C, .last = 0x11D3D, .width = 0 },
    .{ .first = 0x11D3F, .last = 0x11D45, .width = 0 },
    .{ .first = 0x11D47, .last = 0x11D47, .width = 0 },
    .{ .first = 0x11D90, .last = 0x11D91, .width = 0 },
    .{ .first = 0x11D95, .last = 0x11D95, .width = 0 },
    .{ .first = 0x11D97, .last = 0x11D97, .width = 0 },
    .{ .first = 0x11EF3, .last = 0x11EF4, .width = 0 },
    .{ .first = 0x16AF0, .last = 0x16AF4, .width = 0 }, // bassa vah combining
    .{ .first = 0x16B30, .last = 0x16B36, .width = 0 }, // pahawh hmong
    .{ .first = 0x16F4F, .last = 0x16F4F, .width = 0 }, // miao
    .{ .first = 0x16F8F, .last = 0x16F92, .width = 0 },
    .{ .first = 0x16FE4, .last = 0x16FE4, .width = 0 },
    .{ .first = 0x1BC9D, .last = 0x1BC9E, .width = 0 }, // duployan
    .{ .first = 0x1BCA0, .last = 0x1BCA3, .width = 0 }, // shorthand format
    .{ .first = 0x1D167, .last = 0x1D169, .width = 0 }, // musical combining
    .{ .first = 0x1D173, .last = 0x1D182, .width = 0 },
    .{ .first = 0x1D185, .last = 0x1D18B, .width = 0 },
    .{ .first = 0x1D1AA, .last = 0x1D1AD, .width = 0 },
    .{ .first = 0x1D242, .last = 0x1D244, .width = 0 },
    .{ .first = 0x1DA00, .last = 0x1DA36, .width = 0 }, // signwriting
    .{ .first = 0x1DA3B, .last = 0x1DA6C, .width = 0 },
    .{ .first = 0x1DA75, .last = 0x1DA75, .width = 0 },
    .{ .first = 0x1DA84, .last = 0x1DA84, .width = 0 },
    .{ .first = 0x1DA9B, .last = 0x1DA9F, .width = 0 },
    .{ .first = 0x1DAA1, .last = 0x1DAAF, .width = 0 },
    .{ .first = 0x1E000, .last = 0x1E006, .width = 0 }, // glagolitic combining
    .{ .first = 0x1E008, .last = 0x1E018, .width = 0 },
    .{ .first = 0x1E01B, .last = 0x1E021, .width = 0 },
    .{ .first = 0x1E023, .last = 0x1E024, .width = 0 },
    .{ .first = 0x1E026, .last = 0x1E02A, .width = 0 },
    .{ .first = 0x1E130, .last = 0x1E136, .width = 0 }, // nyiakeng puachue hmong
    .{ .first = 0x1E2AE, .last = 0x1E2AE, .width = 0 },
    .{ .first = 0x1E2EC, .last = 0x1E2EF, .width = 0 },
    .{ .first = 0x1E8D0, .last = 0x1E8D6, .width = 0 },
    .{ .first = 0x1E944, .last = 0x1E94A, .width = 0 },
    .{ .first = 0xE0001, .last = 0xE0001, .width = 0 }, // language tag
    .{ .first = 0xE0020, .last = 0xE007F, .width = 0 }, // tag components
    .{ .first = 0xE0100, .last = 0xE01EF, .width = 0 }, // variation selectors supp
};

/// Double-width codepoint ranges: CJK ideographs, fullwidth forms,
/// Hangul syllables, emoji, and related wide characters.
const double_width_ranges = [_]WidthRange{
    .{ .first = 0x1100, .last = 0x115F, .width = 2 }, // hangul choseong jamo
    .{ .first = 0x231A, .last = 0x231B, .width = 2 }, // watch, hourglass
    .{ .first = 0x2329, .last = 0x232A, .width = 2 }, // angle brackets
    .{ .first = 0x23E9, .last = 0x23F3, .width = 2 }, // media controls
    .{ .first = 0x23F8, .last = 0x23FA, .width = 2 }, // media controls
    .{ .first = 0x25FD, .last = 0x25FE, .width = 2 }, // medium small squares
    .{ .first = 0x2614, .last = 0x2615, .width = 2 }, // umbrella, hot beverage
    .{ .first = 0x2648, .last = 0x2653, .width = 2 }, // zodiac signs
    .{ .first = 0x267F, .last = 0x267F, .width = 2 }, // wheelchair
    .{ .first = 0x2693, .last = 0x2693, .width = 2 }, // anchor
    .{ .first = 0x26A1, .last = 0x26A1, .width = 2 }, // high voltage
    .{ .first = 0x26AA, .last = 0x26AB, .width = 2 }, // circles
    .{ .first = 0x26BD, .last = 0x26BE, .width = 2 }, // soccer, baseball
    .{ .first = 0x26C4, .last = 0x26C5, .width = 2 }, // snowman, sun
    .{ .first = 0x26CE, .last = 0x26CE, .width = 2 }, // ophiuchus
    .{ .first = 0x26D4, .last = 0x26D4, .width = 2 }, // no entry
    .{ .first = 0x26EA, .last = 0x26EA, .width = 2 }, // church
    .{ .first = 0x26F2, .last = 0x26F3, .width = 2 }, // fountain, golf
    .{ .first = 0x26F5, .last = 0x26F5, .width = 2 }, // sailboat
    .{ .first = 0x26FA, .last = 0x26FA, .width = 2 }, // tent
    .{ .first = 0x26FD, .last = 0x26FD, .width = 2 }, // fuel pump
    .{ .first = 0x2702, .last = 0x2702, .width = 2 }, // scissors
    .{ .first = 0x2705, .last = 0x2705, .width = 2 }, // check mark
    .{ .first = 0x2708, .last = 0x270D, .width = 2 }, // airplane..writing hand
    .{ .first = 0x270F, .last = 0x270F, .width = 2 }, // pencil
    .{ .first = 0x2712, .last = 0x2712, .width = 2 }, // black nib
    .{ .first = 0x2714, .last = 0x2714, .width = 2 }, // check mark
    .{ .first = 0x2716, .last = 0x2716, .width = 2 }, // heavy multiplication
    .{ .first = 0x271D, .last = 0x271D, .width = 2 }, // latin cross
    .{ .first = 0x2721, .last = 0x2721, .width = 2 }, // star of david
    .{ .first = 0x2728, .last = 0x2728, .width = 2 }, // sparkles
    .{ .first = 0x2733, .last = 0x2734, .width = 2 }, // asterisks
    .{ .first = 0x2744, .last = 0x2744, .width = 2 }, // snowflake
    .{ .first = 0x2747, .last = 0x2747, .width = 2 }, // sparkle
    .{ .first = 0x274C, .last = 0x274C, .width = 2 }, // cross mark
    .{ .first = 0x274E, .last = 0x274E, .width = 2 }, // cross mark
    .{ .first = 0x2753, .last = 0x2755, .width = 2 }, // question/exclamation
    .{ .first = 0x2757, .last = 0x2757, .width = 2 }, // exclamation
    .{ .first = 0x2763, .last = 0x2764, .width = 2 }, // heart exclamation/heart
    .{ .first = 0x2795, .last = 0x2797, .width = 2 }, // plus/minus/divide
    .{ .first = 0x27A1, .last = 0x27A1, .width = 2 }, // right arrow
    .{ .first = 0x27B0, .last = 0x27B0, .width = 2 }, // curly loop
    .{ .first = 0x27BF, .last = 0x27BF, .width = 2 }, // double curly loop
    .{ .first = 0x2934, .last = 0x2935, .width = 2 }, // arrows
    .{ .first = 0x2B05, .last = 0x2B07, .width = 2 }, // arrows
    .{ .first = 0x2B1B, .last = 0x2B1C, .width = 2 }, // squares
    .{ .first = 0x2B50, .last = 0x2B50, .width = 2 }, // star
    .{ .first = 0x2B55, .last = 0x2B55, .width = 2 }, // circle
    .{ .first = 0x2E80, .last = 0x2E99, .width = 2 }, // CJK radicals supp
    .{ .first = 0x2E9B, .last = 0x2EF3, .width = 2 },
    .{ .first = 0x2F00, .last = 0x2FD5, .width = 2 }, // kangxi radicals
    .{ .first = 0x2FF0, .last = 0x2FFF, .width = 2 }, // ideographic desc chars
    .{ .first = 0x3000, .last = 0x303E, .width = 2 }, // CJK symbols + punctuation
    .{ .first = 0x3041, .last = 0x3096, .width = 2 }, // hiragana
    .{ .first = 0x3099, .last = 0x30FF, .width = 2 }, // hiragana + katakana
    .{ .first = 0x3105, .last = 0x312F, .width = 2 }, // bopomofo
    .{ .first = 0x3131, .last = 0x318E, .width = 2 }, // hangul compat jamo
    .{ .first = 0x3190, .last = 0x31E3, .width = 2 }, // kanbun + CJK strokes
    .{ .first = 0x31F0, .last = 0x321E, .width = 2 }, // katakana phonetic ext
    .{ .first = 0x3220, .last = 0x3247, .width = 2 }, // enclosed CJK letters
    .{ .first = 0x3250, .last = 0x4DBF, .width = 2 }, // CJK compat + ext A
    .{ .first = 0x4E00, .last = 0x9FFF, .width = 2 }, // CJK unified ideographs
    .{ .first = 0xA000, .last = 0xA48C, .width = 2 }, // Yi syllables
    .{ .first = 0xA490, .last = 0xA4C6, .width = 2 }, // Yi radicals
    .{ .first = 0xA960, .last = 0xA97C, .width = 2 }, // hangul jamo ext A
    .{ .first = 0xAC00, .last = 0xD7A3, .width = 2 }, // hangul syllables
    .{ .first = 0xF900, .last = 0xFAFF, .width = 2 }, // CJK compat ideographs
    .{ .first = 0xFE10, .last = 0xFE19, .width = 2 }, // vertical forms
    .{ .first = 0xFE30, .last = 0xFE6B, .width = 2 }, // CJK compat forms
    .{ .first = 0xFF01, .last = 0xFF60, .width = 2 }, // fullwidth forms
    .{ .first = 0xFFE0, .last = 0xFFE6, .width = 2 }, // fullwidth signs
    .{ .first = 0x16FE0, .last = 0x16FE3, .width = 2 }, // ideographic symbols
    .{ .first = 0x17000, .last = 0x187F7, .width = 2 }, // tangut
    .{ .first = 0x18800, .last = 0x18CD5, .width = 2 }, // tangut components
    .{ .first = 0x18D00, .last = 0x18D08, .width = 2 }, // tangut supp
    .{ .first = 0x1AFF0, .last = 0x1AFF3, .width = 2 }, // kana ext B
    .{ .first = 0x1AFF5, .last = 0x1AFFB, .width = 2 },
    .{ .first = 0x1AFFD, .last = 0x1AFFE, .width = 2 },
    .{ .first = 0x1B000, .last = 0x1B122, .width = 2 }, // kana supplement
    .{ .first = 0x1B132, .last = 0x1B132, .width = 2 },
    .{ .first = 0x1B150, .last = 0x1B152, .width = 2 }, // small kana ext
    .{ .first = 0x1B155, .last = 0x1B155, .width = 2 },
    .{ .first = 0x1B164, .last = 0x1B167, .width = 2 },
    .{ .first = 0x1B170, .last = 0x1B2FB, .width = 2 }, // nushu
    .{ .first = 0x1F004, .last = 0x1F004, .width = 2 }, // mahjong tile
    .{ .first = 0x1F0CF, .last = 0x1F0CF, .width = 2 }, // playing card
    .{ .first = 0x1F18E, .last = 0x1F18E, .width = 2 }, // AB button
    .{ .first = 0x1F191, .last = 0x1F19A, .width = 2 }, // squared symbols
    .{ .first = 0x1F1E0, .last = 0x1F1FF, .width = 2 }, // regional indicators
    .{ .first = 0x1F200, .last = 0x1F202, .width = 2 }, // enclosed ideographic
    .{ .first = 0x1F210, .last = 0x1F23B, .width = 2 },
    .{ .first = 0x1F240, .last = 0x1F248, .width = 2 },
    .{ .first = 0x1F250, .last = 0x1F251, .width = 2 },
    .{ .first = 0x1F260, .last = 0x1F265, .width = 2 },
    .{ .first = 0x1F300, .last = 0x1F320, .width = 2 }, // misc symbols + pictographs
    .{ .first = 0x1F32D, .last = 0x1F335, .width = 2 },
    .{ .first = 0x1F337, .last = 0x1F37C, .width = 2 },
    .{ .first = 0x1F37E, .last = 0x1F393, .width = 2 },
    .{ .first = 0x1F3A0, .last = 0x1F3CA, .width = 2 },
    .{ .first = 0x1F3CF, .last = 0x1F3D3, .width = 2 },
    .{ .first = 0x1F3E0, .last = 0x1F3F0, .width = 2 },
    .{ .first = 0x1F3F4, .last = 0x1F3F4, .width = 2 },
    .{ .first = 0x1F3F8, .last = 0x1F43E, .width = 2 },
    .{ .first = 0x1F440, .last = 0x1F440, .width = 2 },
    .{ .first = 0x1F442, .last = 0x1F4FC, .width = 2 },
    .{ .first = 0x1F4FF, .last = 0x1F53D, .width = 2 },
    .{ .first = 0x1F54B, .last = 0x1F54E, .width = 2 },
    .{ .first = 0x1F550, .last = 0x1F567, .width = 2 },
    .{ .first = 0x1F57A, .last = 0x1F57A, .width = 2 },
    .{ .first = 0x1F595, .last = 0x1F596, .width = 2 },
    .{ .first = 0x1F5A4, .last = 0x1F5A4, .width = 2 },
    .{ .first = 0x1F5FB, .last = 0x1F64F, .width = 2 },
    .{ .first = 0x1F680, .last = 0x1F6C5, .width = 2 },
    .{ .first = 0x1F6CC, .last = 0x1F6CC, .width = 2 },
    .{ .first = 0x1F6D0, .last = 0x1F6D2, .width = 2 },
    .{ .first = 0x1F6D5, .last = 0x1F6D7, .width = 2 },
    .{ .first = 0x1F6DC, .last = 0x1F6DF, .width = 2 },
    .{ .first = 0x1F6EB, .last = 0x1F6EC, .width = 2 },
    .{ .first = 0x1F6F4, .last = 0x1F6FC, .width = 2 },
    .{ .first = 0x1F7E0, .last = 0x1F7EB, .width = 2 },
    .{ .first = 0x1F7F0, .last = 0x1F7F0, .width = 2 },
    .{ .first = 0x1F90C, .last = 0x1F93A, .width = 2 },
    .{ .first = 0x1F93C, .last = 0x1F945, .width = 2 },
    .{ .first = 0x1F947, .last = 0x1F9FF, .width = 2 },
    .{ .first = 0x1FA00, .last = 0x1FA53, .width = 2 }, // chess symbols
    .{ .first = 0x1FA60, .last = 0x1FA6D, .width = 2 },
    .{ .first = 0x1FA70, .last = 0x1FA7C, .width = 2 },
    .{ .first = 0x1FA80, .last = 0x1FA88, .width = 2 },
    .{ .first = 0x1FA90, .last = 0x1FABD, .width = 2 },
    .{ .first = 0x1FABF, .last = 0x1FAC5, .width = 2 },
    .{ .first = 0x1FACE, .last = 0x1FADB, .width = 2 },
    .{ .first = 0x1FAE0, .last = 0x1FAE8, .width = 2 },
    .{ .first = 0x1FAF0, .last = 0x1FAF8, .width = 2 },
    .{ .first = 0x20000, .last = 0x2FFFD, .width = 2 }, // CJK ext B..
    .{ .first = 0x30000, .last = 0x3FFFD, .width = 2 }, // CJK ext G..
};

/// Comptime validation: assert table is sorted and non-overlapping.
fn assertSortedDisjoint(comptime ranges: []const WidthRange) void {
    for (0..ranges.len) |i| {
        if (ranges[i].first > ranges[i].last) {
            @compileError("WidthRange has first > last");
        }
        if (i > 0 and ranges[i].first <= ranges[i - 1].last) {
            @compileError("WidthRange table is not sorted or has overlaps");
        }
    }
}

comptime {
    assertSortedDisjoint(&zero_width_ranges);
    assertSortedDisjoint(&double_width_ranges);
}

/// Binary search a sorted WidthRange table for a codepoint.
fn searchRanges(
    comptime ranges: []const WidthRange,
    codepoint: u21,
) ?u2 {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = ranges[mid];
        if (codepoint < r.first) {
            hi = mid;
        } else if (codepoint > r.last) {
            lo = mid + 1;
        } else {
            return r.width;
        }
    }
    return null;
}

/// Decode a single UTF-8 codepoint from a byte slice.
/// Returns the codepoint and byte length consumed,
/// or null for invalid UTF-8.
pub fn decodeUtf8(
    bytes: []const u8,
) ?struct { codepoint: u21, len: u3 } {
    if (bytes.len == 0) return null;

    const b0 = bytes[0];

    // ASCII (1 byte)
    if (b0 < 0x80) {
        return .{ .codepoint = b0, .len = 1 };
    }

    // Determine expected length from leading byte
    if (b0 & 0xE0 == 0xC0) {
        // 2-byte sequence: 110xxxxx 10xxxxxx
        if (bytes.len < 2) return null;
        const b1 = bytes[1];
        if (b1 & 0xC0 != 0x80) return null;
        const cp: u21 = (@as(u21, b0 & 0x1F) << 6) |
            @as(u21, b1 & 0x3F);
        // Reject overlong
        if (cp < 0x80) return null;
        return .{ .codepoint = cp, .len = 2 };
    }

    if (b0 & 0xF0 == 0xE0) {
        // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
        if (bytes.len < 3) return null;
        const b1 = bytes[1];
        const b2 = bytes[2];
        if (b1 & 0xC0 != 0x80 or b2 & 0xC0 != 0x80) return null;
        const cp: u21 = (@as(u21, b0 & 0x0F) << 12) |
            (@as(u21, b1 & 0x3F) << 6) |
            @as(u21, b2 & 0x3F);
        // Reject overlong and surrogates
        if (cp < 0x800 or (cp >= 0xD800 and cp <= 0xDFFF)) {
            return null;
        }
        return .{ .codepoint = cp, .len = 3 };
    }

    if (b0 & 0xF8 == 0xF0) {
        // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        if (bytes.len < 4) return null;
        const b1 = bytes[1];
        const b2 = bytes[2];
        const b3 = bytes[3];
        if (b1 & 0xC0 != 0x80 or
            b2 & 0xC0 != 0x80 or
            b3 & 0xC0 != 0x80) return null;
        const cp: u21 = (@as(u21, b0 & 0x07) << 18) |
            (@as(u21, b1 & 0x3F) << 12) |
            (@as(u21, b2 & 0x3F) << 6) |
            @as(u21, b3 & 0x3F);
        // Reject overlong and out-of-range
        if (cp < 0x10000 or cp > 0x10FFFF) return null;
        return .{ .codepoint = cp, .len = 4 };
    }

    // Invalid leading byte (continuation byte or 0xFE/0xFF)
    return null;
}

/// Return the display width (terminal columns) of a single codepoint.
/// Returns 0 for control chars and combining marks,
/// 1 for most chars, 2 for fullwidth/wide.
pub fn charWidth(codepoint: u21) u2 {
    // Check zero-width first (control, combining, etc.)
    if (searchRanges(&zero_width_ranges, codepoint)) |w| {
        return w;
    }
    // Check double-width (CJK, emoji, fullwidth, etc.)
    if (searchRanges(&double_width_ranges, codepoint)) |w| {
        return w;
    }
    // Everything else is single-width
    return 1;
}

/// Return total display width (terminal columns) for an entire
/// UTF-8 string.
pub fn displayWidth(text: []const u8) usize {
    var width: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (decodeUtf8(text[i..])) |result| {
            width += charWidth(result.codepoint);
            i += result.len;
        } else {
            // Skip invalid byte
            i += 1;
            width += 1;
        }
    }
    return width;
}

/// Truncate a UTF-8 string to fit within max_cols terminal columns.
/// Returns a slice at most max_cols columns wide, never splitting
/// a codepoint.
pub fn truncateToWidth(
    text: []const u8,
    max_cols: usize,
) []const u8 {
    var width: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (decodeUtf8(text[i..])) |result| {
            const cw = charWidth(result.codepoint);
            if (width + cw > max_cols) break;
            width += cw;
            i += result.len;
        } else {
            // Invalid byte treated as 1 column
            if (width + 1 > max_cols) break;
            width += 1;
            i += 1;
        }
    }
    return text[0..i];
}

/// Calculate how many space characters are needed to pad text
/// to target_cols. Returns 0 if text already meets or exceeds
/// target_cols.
pub fn padToWidth(text: []const u8, target_cols: usize) usize {
    const current = displayWidth(text);
    if (current >= target_cols) return 0;
    return target_cols - current;
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

test "ASCII strings are 1 col each" {
    try std.testing.expectEqual(@as(usize, 5), displayWidth("hello"));
    try std.testing.expectEqual(@as(usize, 0), displayWidth(""));
    try std.testing.expectEqual(@as(usize, 1), displayWidth("x"));
}

test "emoji are 2 cols" {
    // Robot face U+1F916: F0 9F A4 96 (4 bytes, 2 cols)
    const robot = "\xF0\x9F\xA4\x96";
    try std.testing.expectEqual(@as(usize, 4), robot.len);
    try std.testing.expectEqual(@as(usize, 2), displayWidth(robot));
}

test "CJK characters are 2 cols" {
    // U+4E2D (chinese "middle"): E4 B8 AD (3 bytes)
    const zhong = "\xE4\xB8\xAD";
    try std.testing.expectEqual(@as(usize, 2), displayWidth(zhong));
    // Two CJK chars = 4 cols
    const two_cjk = "\xE4\xB8\xAD\xE6\x96\x87";
    try std.testing.expectEqual(@as(usize, 4), displayWidth(two_cjk));
}

test "combining marks are 0 cols" {
    // U+0301 combining acute accent: CC 81
    const combining = "\xCC\x81";
    try std.testing.expectEqual(@as(usize, 0), displayWidth(combining));
    // 'a' + combining accent = 1 col total
    const a_accent = "a\xCC\x81";
    try std.testing.expectEqual(@as(usize, 1), displayWidth(a_accent));
}

test "mixed ASCII + emoji + CJK" {
    // "Hi" (2) + robot (2) + zhong (2) + "!" (1) = 7
    const mixed = "Hi\xF0\x9F\xA4\x96\xE4\xB8\xAD!";
    try std.testing.expectEqual(@as(usize, 7), displayWidth(mixed));
}

test "truncateToWidth with emoji at boundary" {
    // "AB" (2 cols) + robot (2 cols) + "CD" (2 cols) = 6 cols total
    const text = "AB\xF0\x9F\xA4\x96CD";
    // Truncate to 3 cols: "AB" fits (2), robot needs 2 more but
    // 2+2=4 > 3, so stop after "AB"
    const result = truncateToWidth(text, 3);
    try std.testing.expectEqualStrings("AB", result);
    // Truncate to 4: "AB" + robot fits
    const result4 = truncateToWidth(text, 4);
    try std.testing.expectEqualStrings(
        "AB\xF0\x9F\xA4\x96",
        result4,
    );
}

test "truncateToWidth with ASCII" {
    try std.testing.expectEqualStrings(
        "hel",
        truncateToWidth("hello", 3),
    );
    try std.testing.expectEqualStrings(
        "hello",
        truncateToWidth("hello", 10),
    );
    try std.testing.expectEqualStrings(
        "",
        truncateToWidth("hello", 0),
    );
}

test "padToWidth calculations" {
    try std.testing.expectEqual(@as(usize, 5), padToWidth("hello", 10));
    try std.testing.expectEqual(@as(usize, 0), padToWidth("hello", 5));
    try std.testing.expectEqual(@as(usize, 0), padToWidth("hello", 3));
    // CJK "zhong" is 2 cols, pad to 5 = 3 spaces
    try std.testing.expectEqual(
        @as(usize, 3),
        padToWidth("\xE4\xB8\xAD", 5),
    );
}

test "decodeUtf8 for 1/2/3/4-byte sequences" {
    // 1-byte: 'A' = 0x41
    const r1 = decodeUtf8("A").?;
    try std.testing.expectEqual(@as(u21, 0x41), r1.codepoint);
    try std.testing.expectEqual(@as(u3, 1), r1.len);

    // 2-byte: U+00E9 (e-acute) = C3 A9
    const r2 = decodeUtf8("\xC3\xA9").?;
    try std.testing.expectEqual(@as(u21, 0xE9), r2.codepoint);
    try std.testing.expectEqual(@as(u3, 2), r2.len);

    // 3-byte: U+4E2D = E4 B8 AD
    const r3 = decodeUtf8("\xE4\xB8\xAD").?;
    try std.testing.expectEqual(@as(u21, 0x4E2D), r3.codepoint);
    try std.testing.expectEqual(@as(u3, 3), r3.len);

    // 4-byte: U+1F916 = F0 9F A4 96
    const r4 = decodeUtf8("\xF0\x9F\xA4\x96").?;
    try std.testing.expectEqual(@as(u21, 0x1F916), r4.codepoint);
    try std.testing.expectEqual(@as(u3, 4), r4.len);
}

test "invalid UTF-8 returns null" {
    // Continuation byte alone
    try std.testing.expect(decodeUtf8("\x80") == null);
    // Truncated 2-byte
    try std.testing.expect(decodeUtf8("\xC3") == null);
    // Truncated 3-byte
    try std.testing.expect(decodeUtf8("\xE4\xB8") == null);
    // Truncated 4-byte
    try std.testing.expect(decodeUtf8("\xF0\x9F\xA4") == null);
    // Overlong 2-byte (encoding U+0000)
    try std.testing.expect(decodeUtf8("\xC0\x80") == null);
    // Invalid bytes 0xFE, 0xFF
    try std.testing.expect(decodeUtf8("\xFE") == null);
    try std.testing.expect(decodeUtf8("\xFF") == null);
}

test "empty string edge cases" {
    try std.testing.expectEqual(@as(usize, 0), displayWidth(""));
    try std.testing.expectEqualStrings(
        "",
        truncateToWidth("", 10),
    );
    try std.testing.expectEqual(@as(usize, 5), padToWidth("", 5));
    try std.testing.expect(decodeUtf8("") == null);
}

test "zero-width chars are 0 cols" {
    // Zero-width space U+200B: E2 80 8B
    const zwsp = "\xE2\x80\x8B";
    try std.testing.expectEqual(@as(usize, 0), displayWidth(zwsp));
    // BOM U+FEFF: EF BB BF
    const bom = "\xEF\xBB\xBF";
    try std.testing.expectEqual(@as(usize, 0), displayWidth(bom));
}

test "charWidth individual codepoints" {
    // ASCII
    try std.testing.expectEqual(@as(u2, 1), charWidth('A'));
    // Control
    try std.testing.expectEqual(@as(u2, 0), charWidth(0x00));
    try std.testing.expectEqual(@as(u2, 0), charWidth(0x1F));
    // CJK
    try std.testing.expectEqual(@as(u2, 2), charWidth(0x4E2D));
    // Fullwidth exclamation
    try std.testing.expectEqual(@as(u2, 2), charWidth(0xFF01));
    // Hangul syllable
    try std.testing.expectEqual(@as(u2, 2), charWidth(0xAC00));
    // Combining diacritical
    try std.testing.expectEqual(@as(u2, 0), charWidth(0x0300));
    // Emoji (robot)
    try std.testing.expectEqual(@as(u2, 2), charWidth(0x1F916));
}

test "fullwidth forms are 2 cols" {
    // Fullwidth Latin A (U+FF21): EF BC A1
    const fw_a = "\xEF\xBC\xA1";
    try std.testing.expectEqual(@as(usize, 2), displayWidth(fw_a));
}

test "hangul syllables are 2 cols" {
    // U+AC00 (first hangul syllable): EA B0 80
    const ga = "\xEA\xB0\x80";
    try std.testing.expectEqual(@as(usize, 2), displayWidth(ga));
}

test "variation selectors are 0 cols" {
    // U+FE0F (variation selector-16): EF B8 8F
    const vs16 = "\xEF\xB8\x8F";
    try std.testing.expectEqual(@as(usize, 0), displayWidth(vs16));
}

test "truncateToWidth with CJK at boundary" {
    // Two CJK chars (4 cols) + "a" (1 col) = 5 cols
    const text = "\xE4\xB8\xAD\xE6\x96\x87a";
    // Truncate to 3: first CJK fits (2), second needs 2 more
    // but 2+2=4 > 3, so stop after first
    const result = truncateToWidth(text, 3);
    try std.testing.expectEqualStrings("\xE4\xB8\xAD", result);
}

test {
    std.testing.refAllDecls(@This());
}
