"""Small, reproducible multilingual text preprocessing example.

The implementation intentionally uses only the Python standard library so it
can run offline in a fresh checkout.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


DOCUMENTS: dict[str, str] = {
    "English": (
        "Shaniwar Wada is a magnificent historical fortification located in the city of Pune. "
        "It was constructed in 1732 by Peshwa Baji Rao I as the main seat of the Peshwas. "
        "This palace became the political heart and grand capital of the expanding Maratha Empire. "
        "The architecture displays a brilliant blend of Maratha style and exquisite Mughal design elements. "
        "Its massive foundations consist of strong stone layers while the upper floors were built of timber. "
        "The complex features five grand gateways including the famous Delhi Darwaza facing north. "
        "A spectacular seven-storied structure once stood proudly within the inner courtyard walls. "
        "Unfortunately an unexpected and devastating fire destroyed the entire palace complex in 1828. "
        "Today the surviving stone ruins attract thousands of tourists from all over the world. "
        "It stands as a timeless symbol of Maharashtra's rich historical heritage and pride."
    ),
    "Hindi": (
        "शनिवार वाड़ा पुणे शहर में स्थित एक भव्य ऐतिहासिक किला है। "
        "इसका निर्माण 1732 में पेशवा बाजीराव प्रथम ने मुख्य निवास के रूप में कराया था। "
        "यह महल विशाल मराठा साम्राज्य का प्रमुख राजनीतिक और प्रशासनिक केंद्र बन गया। "
        "इसकी वास्तुकला में मराठा शैली और उत्तम मुगल डिजाइन का सुंदर मिश्रण दिखाई देता है। "
        "किले की मजबूत नींव पत्थरों से बनी है जबकि ऊपरी मंजिलों में लकड़ी का उपयोग हुआ था। "
        "इस परिसर में पांच बड़े मुख्य द्वार हैं जिसमें प्रसिद्ध दिल्ली दरवाजा उत्तर की ओर है। "
        "एक समय इस आंतरिक प्रांगण के भीतर एक शानदार सात मंजिला इमारत खड़ी थी। "
        "दुर्भाग्य से 1828 में एक विनाशकारी आग ने पूरे महल परिसर को नष्ट कर दिया। "
        "आज इसके बचे हुए ऐतिहासिक अवशेष दुनिया भर के हजारों पर्यटकों को आकर्षित करते हैं। "
        "यह किला महाराष्ट्र के समृद्ध इतिहास और गौरव का एक कालातीत प्रतीक माना जाता है।"
    ),
    "Marathi": (
        "शनिवार वाडा हा पुणे शहरातील एक अत्यंत भव्य आणि ऐतिहासिक किल्ला आहे. "
        "या वास्तूची उभारणी 1732 मध्ये पेशवे बाजीराव पहिले यांनी मुख्य निवासस्थान म्हणून केली. "
        "हा राजवाडा बलाढ्य मराठा साम्राज्याचे प्रमुख राजकीय आणि प्रशासकीय केंद्र बनला. "
        "याच्या वास्तुकलेमध्ये मराठा साम्राज्य शैली आणि उत्कृष्ट मुघल डिझाइनचे सुंदर मिश्रण दिसून येते. "
        "किल्ल्याचा पाया मजबूत दगडांचा बनवला असून वरील मजले लाकडाचे बांधण्यात आले होते. "
        "या ऐतिहासिक वास्तूला एकूण पाच मोठे दरवाजे असून प्रसिद्ध दिल्ली दरवाजा उत्तरेकडे तोंड करून आहे. "
        "एकदा या अंतर्गत प्रांगणात एक भव्य आणि देखणी सात मजली इमारत उभी होती. "
        "दुर्दैवाने 1828 मध्ये लागलेल्या एका भीषण आगीत संपूर्ण राजवाडा परिसर नष्ट झाला. "
        "आज या किल्ल्याचे अवशेष जगभरातील हजारो पर्यटकांना पुण्याकडे आकर्षित करतात. "
        "हा वाडा महाराष्ट्राच्या समृद्ध ऐतिहासिक वारशाचे आणि अस्मितेचे एक जिवंत प्रतीक आहे."
    ),
}

ENGLISH_STOPWORDS = {
    "a", "an", "and", "as", "by", "from", "in", "is", "it", "of", "on",
    "the", "this", "to", "was", "were", "while", "with", "its", "once",
}
HINDI_STOPWORDS = {"में", "एक", "है", "ने", "के", "का", "और", "की", "इसकी", "से", "बनी", "जबकि", "हुआ", "इस", "हैं", "जिसमें", "ओर", "इसके", "को", "माना", "था", "के", "रूप"}
MARATHI_STOPWORDS = {"हा", "एक", "आहे", "या", "मध्ये", "म्हणून", "केली", "झाला", "याच्या", "चा", "असून", "होते", "एकूण", "करून", "होती", "लागलेल्या", "एका", "वाडा", "चे", "चा"}


@dataclass
class LanguageResult:
    language: str
    raw_tokens: list[str]
    filtered_tokens: list[str]
    processed_tokens: list[str]

    @property
    def processed_text(self) -> str:
        return " ".join(self.processed_tokens)


def hindi_stemmer(word: str) -> str:
    """Remove common Hindi inflectional suffixes without breaking short words."""
    suffixes = "ाओं ाएं ाइयों ाएँ ियों ओंग ावर ाव ान ाश कर ाओ िए ाई ाँ ो े ी ा करो वाला वाली वाले पन ता ने ना नी से को का की के में पर गा गी गे या यी ये ं ों याँ ियाँ ए ओ उ ई आ".split()
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[:-len(suffix)]
    return word


def _tokens(text: str) -> list[str]:
    # Split on punctuation and whitespace while preserving Indic combining marks.
    pieces = re.split(r"[\s।,.!?;:()\[\]{}\"“”‘’']+", text.lower())
    return [piece for piece in pieces if piece]


def _english_stem(word: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def process_documents(documents: Mapping[str, str] = DOCUMENTS) -> dict[str, LanguageResult]:
    """Tokenize, filter, and normalize the supplied multilingual documents."""
    results: dict[str, LanguageResult] = {}
    stopwords = {"English": ENGLISH_STOPWORDS, "Hindi": HINDI_STOPWORDS, "Marathi": MARATHI_STOPWORDS}
    for language, text in documents.items():
        raw = _tokens(text)
        filtered = [token for token in raw if token not in stopwords.get(language, set())]
        if language == "English":
            processed = [_english_stem(token) for token in filtered]
        elif language == "Hindi":
            processed = [hindi_stemmer(token) for token in filtered]
        else:
            processed = filtered
        results[language] = LanguageResult(language, raw, filtered, processed)
    return results


def summary(results: Mapping[str, LanguageResult]) -> list[dict[str, int | str]]:
    return [{"Language": result.language, "Input Token Count": len(result.raw_tokens), "Output Token Count": len(result.processed_tokens), "Unique Output Tokens": len(set(result.processed_tokens))} for result in results.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multilingual IR preprocessing demo.")
    parser.add_argument("--json", type=Path, help="Write full processed output to a JSON file.")
    args = parser.parse_args()
    results = process_documents()
    for row in summary(results):
        print(f"{row['Language']}: input={row['Input Token Count']} tokens -> output={row['Output Token Count']} tokens ({row['Unique Output Tokens']} unique)")
    if args.json:
        args.json.write_text(json.dumps({name: asdict(result) for name, result in results.items()}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()