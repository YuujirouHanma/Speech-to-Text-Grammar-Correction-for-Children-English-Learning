---
license: cc-by-nc-sa-4.0
language: en
tags:
  - grammatical-error-correction
  - t5
  - ctranslate2
---

# SpeakFix T5 grammar correction (v2, CTranslate2)

A T5 model fine-tuned to correct English grammar and converted to CTranslate2 with int8 weights.
It is the corrector of [SpeakFix](https://github.com/YuujirouHanma/Speech-to-Text-Grammar-Correction-for-Children-English-Learning).

**Training data:** JFLEG (CC BY-NC-SA 4.0) and the Kaggle "Grammar Correction" dataset. Because of
JFLEG's license, this model is for non-commercial use.

**Results with SpeakFix's confidence filter (margin 2.5), on the JFLEG test set of 747 sentences:**
BLEU with 4 references 0.871, GLEU 0.545, ERRANT F0.5 0.639, exact match 35.7%.

**Limits:** it is meant to be used with the filter above, which holds back about half of the
suggestions for wrong sentences in the test data. It has not been evaluated on children's speech.
