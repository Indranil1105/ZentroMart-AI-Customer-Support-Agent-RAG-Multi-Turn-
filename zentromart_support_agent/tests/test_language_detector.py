"""Unit tests for the language detection module.

Validates language detection for:
    - en (English)
    - hi (Hindi / Devanagari)
    - hi-en (Hinglish / Latin script Hindi)
    - mr (Marathi / Devanagari with Marathi keywords or Latin Marathi)
    - ta (Tamil)
    - kn (Kannada)
    - te (Telugu)
    - ml (Malayalam)
    - bn (Bengali)
    - gu (Gujarati)
Along with edge cases and end-to-end dataset validation on questions.csv,
conversations.csv, and support_logs.csv.
"""

from pathlib import Path
import csv
import unittest

from zentromart_support_agent.agent.language_detector import (
    SUPPORTED_LANGUAGES,
    LANGUAGE_NAMES,
    detect_language,
    get_indic_script_counts,
    is_marathi_devanagari,
    is_marathi_latin,
    is_hinglish,
)

DATASET_ROOT = Path(r"c:\Users\Dell\Downloads\Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026")


class TestSupportedLanguagesConstants(unittest.TestCase):
    """Validate exported language definitions and display names."""

    def test_supported_languages_list(self):
        expected = {"en", "hi", "hi-en", "ta", "kn", "te", "ml", "bn", "gu", "mr"}
        self.assertEqual(set(SUPPORTED_LANGUAGES), expected)
        self.assertEqual(len(SUPPORTED_LANGUAGES), 10)

    def test_language_names_mapping(self):
        for code in SUPPORTED_LANGUAGES:
            self.assertIn(code, LANGUAGE_NAMES)
            self.assertIsInstance(LANGUAGE_NAMES[code], str)
            self.assertGreater(len(LANGUAGE_NAMES[code]), 0)


class TestEnglishDetection(unittest.TestCase):
    """Test standard English inputs including brand names and numbers."""

    def test_english_queries(self):
        queries = [
            "How fast are refunds to the ZentroMart Wallet?",
            "Can I schedule a delivery at midnight?",
            "Is there a student discount?",
            "What happens if I'm not home when the courier comes?",
            "My refund to credit card was started on 17 October 2026. By which date should I have it?",
            "There's an order on my account that I never placed.",
            "The Kestrel Book 14 Laptop (SKU ELE-0023) I received yesterday has a crack.",
            "The order number is 6649 2475.",
            "The last four digits are 6073.",
            "I opened the Neelam Eau de Parfum 50 ml and tried it once. Can I still return it?",
            "My Bluetooth speaker won't turn on at all.",
            "Tell me the phone number of the person who ordered to my building yesterday.",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(detect_language(q), "en")


class TestHindiDevanagariDetection(unittest.TestCase):
    """Test Hindi queries written in Devanagari script."""

    def test_hindi_devanagari_queries(self):
        queries = [
            "मैंने Gulmohar Matte Lipstick खोल लिया है, क्या अब भी वापस कर सकती हूँ?",
            "मुझे Pagecraft A5 Ruled Notebook (Pack of 4) लौटाने के लिए कितने दिन मिलेंगे?",
            "कितने दिन में वापस करना होगा?",
            "मुझे Sootra Canvas Tote Bag लौटाने के लिए कितने दिन मिलेंगे?",
            "क्या आपकी कोई दुकान कोच्चि में है?",
            "ऑर्डर शिप हो गया है, क्या अब कैंसिल हो सकता है?",
            "मुझे Aangan Cotton Double Bedsheet Set लौटाने के लिए कितने दिन मिलेंगे?",
            "मेरा सामान टूटा हुआ आया है, अब क्या करूँ?",
            "नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।",
            "ऑर्डर नंबर है 7936 3550।",
            "आख़िरी चार अंक 2949 हैं।",
            "क्या मैं UrbanThread Slim Fit Jeans वापस कर सकती हूँ?",
            "और रिफ़ंड कितने दिन में आएगा?",
            "क्या मैं किसी इंसान से बात कर सकती हूँ?",
            "मेरा रिफ़ंड कहाँ है?",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(detect_language(q), "hi")


class TestHinglishDetection(unittest.TestCase):
    """Test Hinglish (Latin-script Hindi words mixed with English)."""

    def test_hinglish_queries(self):
        queries = [
            "SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?",
            "credit card wala refund 17 October 2026 ko shuru hua, kab tak aa jaana chahiye?",
            "Bulk order pe GST invoice milega kya?",
            "UPI wala refund 12 October 2026 ko shuru hua, kab tak aa jaana chahiye?",
            "net banking wala refund 12 October 2026 ko shuru hua, kab tak aa jaana chahiye?",
            "Sale mein 10 October ko order kiya tha (UrbanThread Slim Fit Jeans), return ke kitne din hain?",
            "Mera order (Kestrel K7 Smartphone 256 GB) parso deliver hua, kitne din tak return kar sakti hoon?",
            "Hello, mujhe order ke baare mein help chahiye.",
            "Order number hai 8978 2837.",
            "Last four digits 4533 hain.",
            "Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?",
            "Kya main Kaafi Running Sneakers return kar sakti hoon?",
            "Aur refund kitne din mein aayega?",
            "Mera refund kahan hai?",
            "Theek hai. Ek aur sawaal: agle order pe do coupon laga sakta hoon?",
            "Achha. Aur mera order kab ship hua tha?",
            "Order late aaya, mujhe 300 rupaye ka voucher chahiye.",
            "Please ise cancel kar do.",
            "Mujhe ise cancel karna hai.",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(detect_language(q), "hi-en")


class TestMarathiDetection(unittest.TestCase):
    """Test Marathi queries in Devanagari and Latin script."""

    def test_marathi_queries(self):
        queries = [
            # Devanagari with Marathi keywords
            "माझा ऑर्डर कुठे आहे?",
            "मला मदत हवी आहे",
            "हे कसा करायचा?",
            "डिलिव्हरी कधी होणार आहे?",
            "ऑर्डर रद्द करा आणि पैसे परत द्या",
            "मला हे उत्पादन पाहिजे",
            "तुमचा पत्ता काय आहे?",
            "रिफंड कधी मिळेल?",
            "काही अडचण नाही",
            # Devanagari with Marathi distinctive character ळ (LLA)
            "माझा वेगळा पत्ता आहे",
            "वेळ लागेल का?",
            "जवळचे केंद्र कुठे आहे?",
            # Latin script Marathi keywords as specified in requirements
            "aahe mala kasa",
            "mala order cancel karaycha aahe",
            "kasa aahe ha product",
            "majha order kadhi yenar aahe",
            "tumcha delivery boy kadhi yenar aahe",
            "mala he parat karayche aahe",
        ]
        for q in queries:
            with self.subTest(query=q):
                self.assertEqual(detect_language(q), "mr")


class TestDravidianAndOtherIndicLanguages(unittest.TestCase):
    """Test Tamil, Kannada, Telugu, Malayalam, Bengali, Gujarati queries."""

    def test_tamil_detection(self):
        self.assertEqual(detect_language("என் ஆர்டர் எங்கே?"), "ta")
        self.assertEqual(detect_language("வணக்கம், எனக்கு உதவி வேண்டும்"), "ta")
        self.assertEqual(detect_language("UrbanThread ஜீன்ஸ் கிழிந்துவிட்டது"), "ta")

    def test_kannada_detection(self):
        self.assertEqual(detect_language("ನನ್ನ ಆದೇಶ ಎಲ್ಲಿದೆ?"), "kn")
        self.assertEqual(detect_language("ನನಗೆ ಸಹಾಯ ಬೇಕು"), "kn")
        self.assertEqual(detect_language("ಆರ್ಡರ್ ತಲುಪಿಸಲು ಎಷ್ಟು ದಿನ ಬೇಕು?"), "kn")

    def test_telugu_detection(self):
        self.assertEqual(detect_language("నా ఆర్డర్ ఎక్కడ ఉంది?"), "te")
        self.assertEqual(detect_language("నాకు సహాయం కావాలి"), "te")
        self.assertEqual(detect_language("రీఫండ్ ఎప్పుడు వస్తుంది?"), "te")

    def test_malayalam_detection(self):
        self.assertEqual(detect_language("എന്റെ ഓർഡർ എവിടെയാണ്?"), "ml")
        self.assertEqual(detect_language("എനിക്ക് സഹായം വേണം"), "ml")
        self.assertEqual(detect_language("റീഫണ്ട് എപ്പോൾ ലഭിക്കും?"), "ml")

    def test_bengali_detection(self):
        self.assertEqual(detect_language("আমার অর্ডার কোথায়?"), "bn")
        self.assertEqual(detect_language("আমার সাহায্য দরকার"), "bn")
        self.assertEqual(detect_language("রিফান্ড কখন আসবে?"), "bn")

    def test_gujarati_detection(self):
        self.assertEqual(detect_language("મારો ઓર્ડર ક્યાં છે?"), "gu")
        self.assertEqual(detect_language("મને મદદ જોઈએ છે"), "gu")
        self.assertEqual(detect_language("રિફંડ ક્યારે આવશે?"), "gu")


class TestEdgeCasesAndDefensiveBehavior(unittest.TestCase):
    """Test non-string, empty, whitespace, and special character inputs."""

    def test_empty_and_whitespace(self):
        for empty_val in ["", "   ", "\t\n", " \n "]:
            with self.subTest(val=repr(empty_val)):
                self.assertEqual(detect_language(empty_val), "en")

    def test_non_string_types(self):
        for non_string in [None, 12345, 99.9, [], {}, True, False]:
            with self.subTest(val=repr(non_string)):
                self.assertEqual(detect_language(non_string), "en")

    def test_numbers_and_punctuation(self):
        self.assertEqual(detect_language("12345678"), "en")
        self.assertEqual(detect_language("??? !!! --- @@@"), "en")
        self.assertEqual(detect_language("100% #1"), "en")

    def test_helper_functions(self):
        counts = get_indic_script_counts("नमस्ते hello உலகம்")
        self.assertGreater(counts["devanagari"], 0)
        self.assertGreater(counts["ta"], 0)
        self.assertEqual(counts["kn"], 0)

        self.assertTrue(is_marathi_devanagari("माझा ऑर्डर कुठे आहे?"))
        self.assertFalse(is_marathi_devanagari("मेरा ऑर्डर कहाँ है?"))
        self.assertTrue(is_marathi_latin("aahe mala kasa"))
        self.assertTrue(is_hinglish("order number hai 1234"))
        self.assertFalse(is_hinglish("order number is 1234"))


class TestPracticeAndRealDatasetAccuracy(unittest.TestCase):
    """Validate 100% detection accuracy on the capstone project dataset files."""

    def test_questions_csv_accuracy(self):
        path = DATASET_ROOT / "practice" / "questions.csv"
        self.assertTrue(path.exists(), f"Missing dataset file: {path}")

        total = 0
        correct = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                gold_lang = row["language"].strip()
                question = row["question"]
                pred = detect_language(question)
                self.assertEqual(
                    pred,
                    gold_lang,
                    f"Mismatch in questions.csv row {row.get('id')}: "
                    f"expected '{gold_lang}', got '{pred}' for text: {question}",
                )
                correct += 1

        self.assertEqual(total, 52)
        self.assertEqual(correct, 52)

    def test_conversations_csv_accuracy(self):
        path = DATASET_ROOT / "practice" / "conversations.csv"
        self.assertTrue(path.exists(), f"Missing dataset file: {path}")

        total = 0
        correct = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                gold_lang = row["language"].strip()
                msg = row["customer_message"]
                pred = detect_language(msg)
                self.assertEqual(
                    pred,
                    gold_lang,
                    f"Mismatch in conversations.csv row {row.get('conversation_id')}: "
                    f"expected '{gold_lang}', got '{pred}' for text: {msg}",
                )
                correct += 1

        self.assertEqual(total, 96)
        self.assertEqual(correct, 96)

    def test_support_logs_csv_accuracy(self):
        path = DATASET_ROOT / "data" / "support_logs.csv"
        self.assertTrue(path.exists(), f"Missing dataset file: {path}")

        total = 0
        correct = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                gold_lang = row["language"].strip()
                transcript = row["transcript"]
                pred = detect_language(transcript)
                self.assertEqual(
                    pred,
                    gold_lang,
                    f"Mismatch in support_logs.csv log_id {row.get('log_id')}: "
                    f"expected '{gold_lang}', got '{pred}'",
                )
                correct += 1

        self.assertEqual(total, 244)
        self.assertEqual(correct, 244)


if __name__ == "__main__":
    unittest.main()
