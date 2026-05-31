"""
Grade 1 Braille lookup table.
Dot positions in a Braille cell (2 cols x 3 rows):
  1 4
  2 5
  3 6
Each pattern is a frozenset of active dot positions (1-indexed).
"""

# --- Letters ---
BRAILLE_TO_CHAR = {
    frozenset([1]):             'a',
    frozenset([1, 2]):          'b',
    frozenset([1, 4]):          'c',
    frozenset([1, 4, 5]):       'd',
    frozenset([1, 5]):          'e',
    frozenset([1, 2, 4]):       'f',
    frozenset([1, 2, 4, 5]):    'g',
    frozenset([1, 2, 5]):       'h',
    frozenset([2, 4]):          'i',
    frozenset([2, 4, 5]):       'j',
    frozenset([1, 3]):          'k',
    frozenset([1, 2, 3]):       'l',
    frozenset([1, 3, 4]):       'm',
    frozenset([1, 3, 4, 5]):    'n',
    frozenset([1, 3, 5]):       'o',
    frozenset([1, 2, 3, 4]):    'p',
    frozenset([1, 2, 3, 4, 5]): 'q',
    frozenset([1, 2, 3, 5]):    'r',
    frozenset([2, 3, 4]):       's',
    frozenset([2, 3, 4, 5]):    't',
    frozenset([1, 3, 6]):       'u',
    frozenset([1, 2, 3, 6]):    'v',
    frozenset([2, 4, 5, 6]):    'w',
    frozenset([1, 3, 4, 6]):    'x',
    frozenset([1, 3, 4, 5, 6]): 'y',
    frozenset([1, 3, 5, 6]):    'z',

    # Space
    frozenset():                ' ',

    # Numbers (preceded by number indicator dots 3,4,5,6)
    # We handle the number indicator separately in the pipeline

    # Common punctuation
    frozenset([2]):             ',',
    frozenset([2, 5]):          ':',
    frozenset([2, 3]):          ';',
    frozenset([2, 5, 6]):       '.',
    frozenset([2, 3, 5]):       '!',
    frozenset([2, 3, 6]):       '"',
    frozenset([3, 5]):          '?',
    frozenset([3, 6]):          '-',
    frozenset([1, 2, 3, 4, 5, 6]): '#',  # number indicator
    frozenset([3, 4, 5, 6]):    '#',     # also used as number indicator
    frozenset([6]):             ' ',     # capital indicator (we skip capitalization for speed)
}

# Number mode map (same dot patterns as a-j but represent 1-0)
NUMBER_MAP = {
    frozenset([1]):             '1',
    frozenset([1, 2]):          '2',
    frozenset([1, 4]):          '3',
    frozenset([1, 4, 5]):       '4',
    frozenset([1, 5]):          '5',
    frozenset([1, 2, 4]):       '6',
    frozenset([1, 2, 4, 5]):    '7',
    frozenset([1, 2, 5]):       '8',
    frozenset([2, 4]):          '9',
    frozenset([2, 4, 5]):       '0',
}

NUMBER_INDICATOR = frozenset([3, 4, 5, 6])
CAPITAL_INDICATOR = frozenset([6])


def pattern_to_char(pattern: frozenset, number_mode: bool = False) -> str:
    """Convert a dot pattern (frozenset of active dot positions) to a character."""
    if number_mode:
        return NUMBER_MAP.get(pattern, '?')
    return BRAILLE_TO_CHAR.get(pattern, '?')


def decode_cells(cell_patterns: list) -> str:
    """
    Decode a list of Braille cell patterns into English text.
    Handles number indicator and capital indicator.
    """
    result = []
    number_mode = False
    capital_next = False

    for pattern in cell_patterns:
        if pattern == NUMBER_INDICATOR:
            number_mode = True
            continue
        if pattern == CAPITAL_INDICATOR:
            capital_next = True
            continue
        if pattern == frozenset():
            number_mode = False
            result.append(' ')
            continue

        if number_mode:
            ch = NUMBER_MAP.get(pattern, '?')
        else:
            ch = BRAILLE_TO_CHAR.get(pattern, '?')

        if capital_next and ch.isalpha():
            ch = ch.upper()
            capital_next = False

        result.append(ch)

    return ''.join(result)
