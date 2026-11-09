//! Text normalization before WER/CER scoring.
//!
//! ASR scores are only comparable under a fixed normalization. The default
//! [`Normalizer`] applies, in order:
//!
//! 1. **Numbers to words** ([`NumberHandling::Words`]): a run of ASCII digits
//!    that is not glued to letters becomes English words. Thousands
//!    separators (`1,000`), a decimal part (`3.14` → `three point one four`,
//!    digits read one by one), a leading minus after whitespace or at the
//!    start (`-5` → `minus five`) and ordinal suffixes (`21st` → `twenty
//!    first`) are understood. Integers are read without "and" (`123` → `one
//!    hundred twenty three`) up to 999 999 999 999 999; longer runs are
//!    read digit by digit. Digits inside words (`mp3`) are left alone.
//! 2. **Lowercasing** (Unicode `to_lowercase`).
//! 3. **Punctuation removal**: every character that is neither alphanumeric
//!    nor whitespace becomes a space, except apostrophes (`'` or `’`, written
//!    as `'`) between two alphanumeric characters when
//!    [`Normalizer::keep_apostrophes`] is set (`what's` stays one word).
//!    Hyphens therefore split words (`mm-hmm` → `mm hmm`).
//! 4. **Whitespace**: runs of whitespace become one space; leading and
//!    trailing whitespace is dropped. This step always applies.
//!
//! [`Normalizer::none`] only applies step 4.

/// How digit runs are treated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NumberHandling {
    /// Leave digits as they are.
    Keep,
    /// Spell numbers out in English words (see the module docs).
    #[default]
    Words,
}

/// Configurable text normalizer (see the [module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Normalizer {
    /// Lowercase everything.
    pub lowercase: bool,
    /// Replace punctuation and symbols with spaces.
    pub remove_punctuation: bool,
    /// Keep apostrophes inside words when removing punctuation.
    pub keep_apostrophes: bool,
    /// Number handling.
    pub numbers: NumberHandling,
}

impl Default for Normalizer {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            keep_apostrophes: true,
            numbers: NumberHandling::Words,
        }
    }
}

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [(u64, &str); 4] = [
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
];

fn below_thousand(n: u64, out: &mut Vec<&'static str>) {
    debug_assert!(n < 1000);
    if n >= 100 {
        out.push(ONES[(n / 100) as usize]);
        out.push("hundred");
    }
    let r = n % 100;
    if r >= 20 {
        out.push(TENS[(r / 10) as usize]);
        if !r.is_multiple_of(10) {
            out.push(ONES[(r % 10) as usize]);
        }
    } else if r > 0 || n == 0 {
        out.push(ONES[r as usize]);
    }
}

/// English words for an integer: `123` → `one hundred twenty three`.
pub fn integer_words(n: u64) -> String {
    if n == 0 {
        return "zero".into();
    }
    let mut parts: Vec<String> = Vec::new();
    let mut rest = n;
    for (scale, name) in SCALES {
        if rest >= scale {
            let chunk = rest / scale;
            if chunk >= 1000 {
                // Beyond the largest scale ("one thousand trillion").
                parts.push(integer_words(chunk));
            } else {
                let mut w = Vec::new();
                below_thousand(chunk, &mut w);
                parts.push(w.join(" "));
            }
            parts.push(name.to_string());
            rest %= scale;
        }
    }
    if rest > 0 {
        let mut w = Vec::new();
        below_thousand(rest, &mut w);
        parts.push(w.join(" "));
    }
    parts.join(" ")
}

/// The ordinal form of a cardinal's last word: `twenty one` → `twenty first`.
fn ordinal(cardinal: &str) -> String {
    let (head, last) = match cardinal.rsplit_once(' ') {
        Some((h, l)) => (format!("{h} "), l),
        None => (String::new(), cardinal),
    };
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        l if l.ends_with('y') => format!("{}ieth", &l[..l.len() - 1]),
        l => format!("{l}th"),
    };
    format!("{head}{last}")
}

fn digits_words(digits: &str) -> String {
    digits
        .bytes()
        .map(|d| ONES[(d - b'0') as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

/// Largest integer read as words; longer runs are read digit by digit.
const MAX_WORDS: u64 = 999_999_999_999_999;

/// Spell out numbers in `text` (step 1 of the module docs).
fn numbers_to_words(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let prev_alnum = i > 0 && chars[i - 1].is_alphanumeric();
        if !c.is_ascii_digit() || prev_alnum {
            // Copy a whole word with embedded digits (e.g. `mp3`) verbatim.
            if c.is_alphanumeric() {
                while i < chars.len() && chars[i].is_alphanumeric() {
                    out.push(chars[i]);
                    i += 1;
                }
            } else {
                out.push(c);
                i += 1;
            }
            continue;
        }
        // Integer part, with optional `,ddd` groups after a 1-3 digit lead.
        let start = i;
        let mut int = String::new();
        while i < chars.len() && chars[i].is_ascii_digit() {
            int.push(chars[i]);
            i += 1;
        }
        if int.len() <= 3 {
            while chars.get(i) == Some(&',')
                && i + 3 < chars.len()
                && chars[i + 1..i + 4].iter().all(char::is_ascii_digit)
                && chars.get(i + 4).is_none_or(|d| !d.is_ascii_digit())
            {
                int.extend(&chars[i + 1..i + 4]);
                i += 4;
            }
        }
        // Decimal part.
        let mut frac = String::new();
        if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                frac.push(chars[i]);
                i += 1;
            }
        }
        // Ordinal suffix (only directly after an integer).
        let mut is_ordinal = false;
        if frac.is_empty() && i + 1 < chars.len() {
            let suffix: String = chars[i..i + 2].iter().collect::<String>().to_lowercase();
            let after_alnum = chars.get(i + 2).is_some_and(|c| c.is_alphanumeric());
            if matches!(suffix.as_str(), "st" | "nd" | "rd" | "th") && !after_alnum {
                is_ordinal = true;
                i += 2;
            }
        }
        // Glued to letters (e.g. `3d`, `4k`): leave untouched.
        if !is_ordinal && chars.get(i).is_some_and(|c| c.is_alphabetic()) {
            out.extend(&chars[start..i]);
            continue;
        }
        // A minus sign directly before, itself at the start or after
        // whitespace (`3-4` is a range, not a negative number). It was
        // copied to `out` as the previous character.
        if start > 0 && chars[start - 1] == '-' && (start == 1 || chars[start - 2].is_whitespace())
        {
            out.pop();
            out.push_str("minus");
        }
        let trimmed = int.trim_start_matches('0');
        let int_part = if trimmed.is_empty() {
            "zero".to_string()
        } else {
            match trimmed.parse::<u64>() {
                Ok(n) if n <= MAX_WORDS => integer_words(n),
                _ => digits_words(&int),
            }
        };
        let mut words = if is_ordinal {
            ordinal(&int_part)
        } else {
            int_part
        };
        if !frac.is_empty() {
            words.push_str(" point ");
            words.push_str(&digits_words(&frac));
        }
        out.push(' ');
        out.push_str(&words);
        out.push(' ');
    }
    out
}

impl Normalizer {
    /// Whitespace collapsing only: case, punctuation and digits are kept.
    pub fn none() -> Self {
        Self {
            lowercase: false,
            remove_punctuation: false,
            keep_apostrophes: true,
            numbers: NumberHandling::Keep,
        }
    }

    /// Normalize `text` (see the [module docs](self)).
    pub fn normalize(&self, text: &str) -> String {
        let text = match self.numbers {
            NumberHandling::Words => numbers_to_words(text),
            NumberHandling::Keep => text.to_string(),
        };
        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text
        };
        let text = if self.remove_punctuation {
            let chars: Vec<char> = text.chars().collect();
            let mut out = String::with_capacity(text.len());
            for (i, &c) in chars.iter().enumerate() {
                if c.is_alphanumeric() || c.is_whitespace() {
                    out.push(c);
                } else if self.keep_apostrophes
                    && (c == '\'' || c == '’')
                    && i > 0
                    && chars[i - 1].is_alphanumeric()
                    && chars.get(i + 1).is_some_and(|n| n.is_alphanumeric())
                {
                    out.push('\'');
                } else {
                    out.push(' ');
                }
            }
            out
        } else {
            text
        };
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// The normalized words of `text`.
    pub fn words(&self, text: &str) -> Vec<String> {
        self.normalize(text)
            .split(' ')
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_in_words() {
        assert_eq!(integer_words(0), "zero");
        assert_eq!(integer_words(7), "seven");
        assert_eq!(integer_words(18), "eighteen");
        assert_eq!(integer_words(40), "forty");
        assert_eq!(integer_words(99), "ninety nine");
        assert_eq!(integer_words(100), "one hundred");
        assert_eq!(integer_words(123), "one hundred twenty three");
        assert_eq!(integer_words(1000), "one thousand");
        assert_eq!(integer_words(2024), "two thousand twenty four");
        assert_eq!(integer_words(1_000_001), "one million one");
        assert_eq!(
            integer_words(3_400_050_017),
            "three billion four hundred million fifty thousand seventeen"
        );
        assert_eq!(
            integer_words(999_999_999_999_999),
            "nine hundred ninety nine trillion nine hundred ninety nine billion nine hundred \
             ninety nine million nine hundred ninety nine thousand nine hundred ninety nine"
        );
    }

    #[test]
    fn ordinals() {
        assert_eq!(ordinal("one"), "first");
        assert_eq!(ordinal("twenty one"), "twenty first");
        assert_eq!(ordinal("twenty"), "twentieth");
        assert_eq!(ordinal("twelve"), "twelfth");
        assert_eq!(ordinal("one hundred"), "one hundredth");
        assert_eq!(ordinal("four"), "fourth");
    }

    #[test]
    fn default_normalization() {
        let n = Normalizer::default();
        assert_eq!(
            n.normalize("  It's 18 degrees, and SUNNY in Paris!  "),
            "it's eighteen degrees and sunny in paris"
        );
        assert_eq!(n.normalize("mm-hmm... okay?"), "mm hmm okay");
        assert_eq!(n.normalize("‘quoted’ don’t"), "quoted don't");
        assert_eq!(n.normalize("1,000 people"), "one thousand people");
        assert_eq!(n.normalize("pi is 3.14"), "pi is three point one four");
        assert_eq!(n.normalize("-5 and 3-4"), "minus five and three four");
        assert_eq!(
            n.normalize("the 21st and 2nd"),
            "the twenty first and second"
        );
        assert_eq!(n.normalize("an mp3 in 4k"), "an mp3 in 4k");
        assert_eq!(n.normalize("007"), "seven");
        assert_eq!(n.normalize("0"), "zero");
        assert_eq!(n.normalize("1,2,3"), "one two three");
        assert_eq!(n.normalize("$5.50"), "five point five zero");
        assert_eq!(
            n.normalize("12345678901234567890"),
            "one two three four five six seven eight nine zero one two three four five six \
             seven eight nine zero"
        );
        assert_eq!(n.normalize(""), "");
        assert_eq!(n.words("Café, NAÏVE"), ["café", "naïve"]);
    }

    #[test]
    fn options() {
        let none = Normalizer::none();
        assert_eq!(none.normalize(" Hello,\tWorld 42 "), "Hello, World 42");
        let keep_digits = Normalizer {
            numbers: NumberHandling::Keep,
            ..Normalizer::default()
        };
        assert_eq!(keep_digits.normalize("Room 101!"), "room 101");
        let no_apostrophes = Normalizer {
            keep_apostrophes: false,
            ..Normalizer::default()
        };
        assert_eq!(no_apostrophes.normalize("what's up"), "what s up");
        let case = Normalizer {
            lowercase: false,
            ..Normalizer::default()
        };
        assert_eq!(case.normalize("Paris, 2 days"), "Paris two days");
    }
}
