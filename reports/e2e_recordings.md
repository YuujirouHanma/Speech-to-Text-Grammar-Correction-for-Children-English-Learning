# End-to-end recordings

Real server, real HTTP requests, 2026-11-02, final code, measured with `scripts/check_api.py`. Source of
the expected sentences: `FP/WHISPER DENGAN INPUT AUDIO.docx`.

## Summary

| | Public demo (`ctranslate2:int8_float32@cuda`, `whisper:small.en`, through the HTTPS tunnel) | Docker contents (`ctranslate2:int8_float32@cpu`, `whisper:base.en`, local) |
|---|---|---|
| Team's 10 sentences: corrected | 9/10 | 9/10 |
| Team's 10 sentences: exactly the Expected Output | 2/10 (old result: 1/10) | 2/10 |
| All 15 recordings: corrected | 12/15 | 12/15 |
| All 15 recordings: exactly the expected sentence | 7/15 | 7/15 |
| Same sentences typed: corrected | 15/15 | 15/15 |
| Same sentences typed: exactly the expected sentence | 8/15 | 9/15 |
| Median / slowest server time | 458 / 1051 ms | 1347 / 1688 ms |

Word distance to the Expected Output over the team's 10 sentences (159 words; lower is
better): uncorrected input 43, old results 17, Speech Grammar 20 (GPU) and
17 (CPU). Both pipelines hear the recording first, so STT errors count here. Ten sentences
with one reference each are too few to rank the two systems: on this sample they are close, and the
held-out test sets in `gec_eval.json` are the better measure.

## Team's 10 sentences (public demo)

| Spoken | Expected Output (team document) | Old result (team document) | Speech Grammar | Word distance old / new |
|---|---|---|---|---|
| Although she studied hardly for the exam but she didn’t passed and she was very disappointing about the results. | Although she studied hard for the exam, she didn't pass and she was very disappointed with the results. | Although she studied hardly for the exam but she didn't pass and she was very disappointed about the results . | Although she studied hardly for the exam, she didn't pass and was very disappointed about the results. | 3 / 3 |
| The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | The people who live in the city have many problems with pollution and traffic that is getting worse. | The people who live in the city have many problems with pollution and traffic that is getting worse. | The people who live in the city have many problems with pollution and traffic that is getting worse. | 0 / 0 |
| He was late because he forget his keys and then he realize that he leave it on the table in the kitchen. | He was late because he forgot his keys and then he realized that he left them on the table in the kitchen. | He was late because he forgot his keys and then realized that he left it on the table in the kitchen . | He was late because he forgot his keys and then realized that he left them on the table in the kitchen. | 2 / 1 |
| In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he said that the company is planning to expand their services in other countries. | In the meeting yesterday, he told that the company are planning to expand their services in other countries. | In the meeting yesterday, he told that the company is planning to expand their services to other countries. | 2 / 2 |
| Many students finds difficult to concentrates in online class because of distractions and lack of motivations. | Many students find it difficult to concentrate in online classes because of distractions and lack of motivation. | Many students find it difficult to concentrate in online classes because of distractions and lack of motivations . | Many students find it difficult to concentrate in online class because of distractions and lack of motivation. | 1 / 1 |
| She don’t has enough experience, so they didn’t hired her for the job. | She doesn't have enough experience, so they didn't hire her for the job. | She don't have enough experience, so they didn't hire her for the job. | She doesn't have enough experience, so they didn't hire her for the job. | 1 / 0 |
| The team were not perform well last game because they didn’t prepared properly. | The team did not perform well last game because they weren't properly prepared. | The team did not perform well last game because they didn't prepare properly . | The team did not perform well last game because they didn't prepare properly. | 3 / 3 |
| I am agree with your opinion, but we needs more evidence before taking decision. | I agree with your opinion, but we need more evidence before making a decision. | I agree with your opinion, but we need more evidence before taking a decision. | I agree with your opinion, but we need more evidence before taking a decision. | 1 / 1 |
| He want to know what do means by artificial intelligences in modern technologies. | He wants to know what is meant by artificial intelligence in modern technologies. | He wants to know what do mean by artificial intelligences in modern technologies . | He wants to know what artificial intelligences in modern technologies mean. | 3 / 5 |
| This kind of mistakes is happens when we not focus during the test. | This kind of mistake happens when we are not focused during the test. | This kind of mistakes happens when we are not focused during the test | This kind of mistakes as happens when we not focus during the test. | 1 / 4 |

## All recordings (public demo)

| Spoken (file name) | Whisper heard | Output | Result | ms |
|---|---|---|---|---|
| Although she studied hardly for the exam but she didn’t passed and she was very disappointing about the results. | Although she studied hardly for the exam but she didn't past and she was very disappointing about the results. | Although she studied hardly for the exam, she didn't pass and was very disappointed about the results. | corrected, differs | 1051 |
| The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | The people who live in the city have many problems with pollution and traffic that is getting worse. | exact | 792 |
| He was late because he forget his keys and then he realize that he leave it on the table in the kitchen. | He was late because he forget his keys and then he realize that he leave it on the table in the kitchen. | He was late because he forgot his keys and then realized that he left them on the table in the kitchen. | corrected, differs | 663 |
| In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he told that the company is planning to expand their services to other countries. | corrected, differs | 653 |
| Many students finds difficult to concentrates in online class because of distractions and lack of motivations. | Many students finds difficult to concentrates in online class because of distractions and lack of motivations. | Many students find it difficult to concentrate in online class because of distractions and lack of motivation. | corrected, differs | 576 |
| She don’t has enough experience, so they didn’t hired her for the job. | She don't has enough experience, so they didn't hired her for the job. | She doesn't have enough experience, so they didn't hire her for the job. | exact | 493 |
| The team were not perform well last game because they didn’t prepared properly. | The team were not perform well last game because they didn't prepared properly. | The team did not perform well last game because they didn't prepare properly. | corrected, differs | 458 |
| I am agree with your opinion, but we needs more evidence before taking decision. | I am agree with your opinion, but we needs more evidence before taking decision. | I agree with your opinion, but we need more evidence before taking a decision. | corrected, differs | 477 |
| He want to know what do means by artificial intelligences in modern technologies. | He want to know what do means by artificial intelligences in modern technologies. | He wants to know what artificial intelligences in modern technologies mean. | corrected, differs | 456 |
| This kind of mistakes is happens when we not focus during the test. | This kind of mistakes as happens when we not focus during the test. | This kind of mistakes as happens when we not focus during the test. | unchanged (uncertain_correction_skipped) | 411 |
| they is here today | They is here today. | They are here today. | exact | 240 |
| she go to school | She go to school. | She goes to school. | exact | 241 |
| He readed the book | He read the book. | He read the book. | exact | 245 |
| She can sings well. | She can sing well. | She can sing well. | exact | 280 |
| She go to school. | She go to school. | She goes to school. | exact | 336 |

## All recordings (Docker contents)

| Spoken (file name) | Whisper heard | Output | Result | ms |
|---|---|---|---|---|
| Although she studied hardly for the exam but she didn’t passed and she was very disappointing about the results. | Although she studied hardly for the exam, but she didn't passed and she was very disappointing about the results. | Although she studied hardly for the exam, she didn't pass and was very disappointed about the results. | corrected, differs | 1688 |
| The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | The peoples who lives in the city has many problem with pollution and traffic that is getting more worst. | The people who live in the city have many problems with pollution and traffic that is getting worse. | exact | 1535 |
| He was late because he forget his keys and then he realize that he leave it on the table in the kitchen. | He was late because he forget his keys and then he realized that he leave it on the table in the kitchen. | He was late because he forgot his keys, and then he realized that he left them on the table in the kitchen. | corrected, differs | 1587 |
| In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he told that the company is planning to expand their services to other countries. | corrected, differs | 1639 |
| Many students finds difficult to concentrates in online class because of distractions and lack of motivations. | Many students find it difficult to concentrate in online class because of distractions and lack of motivations. | Many students find it difficult to concentrate in online class because of distractions and lack of motivation. | corrected, differs | 1373 |
| She don’t has enough experience, so they didn’t hired her for the job. | She don't has enough experience, so they didn't hired her for the job. | She doesn't have enough experience, so they didn't hire her for the job. | exact | 1342 |
| The team were not perform well last game because they didn’t prepared properly. | The team were not perform well last game because they didn't prepare properly. | The team did not perform well last game because they didn't prepare properly. | corrected, differs | 1469 |
| I am agree with your opinion, but we needs more evidence before taking decision. | I am agree with your opinion, but we need more evidence before taking decision. | I agree with your opinion, but we need more evidence before taking a decision. | corrected, differs | 1347 |
| He want to know what do means by artificial intelligences in modern technologies. | He want to know what do means by artificial intelligences in modern technologies. | He wants to know what that means by artificial intelligences in modern technologies. | corrected, differs | 1276 |
| This kind of mistakes is happens when we not focus during the test. | This kind of mistakes as happens when we not focus during the test. | This kind of mistakes as happens when we not focus during the test. | unchanged (uncertain_correction_skipped) | 1409 |
| they is here today | They is here today. | They are here today. | exact | 968 |
| she go to school | She go to school. | She goes to school. | exact | 994 |
| He readed the book | He read the book. | He read the book. | exact | 935 |
| She can sings well. | She can sing well. | She can sing well. | exact | 1079 |
| She go to school. | She go to school. | She goes to school. | exact | 1049 |

## Same sentences typed (public demo)

| Typed | Output | Expected | Exact |
|---|---|---|---|
| Although she studied hardly for the exam but she didn’t passed and she was very disappointing about the results. | Although she studied hardly for the exam, she didn't pass and was very disappointed about the results. | Although she studied hard for the exam, she didn't pass and she was very disappointed with the results. | no |
| The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | The people who live in the city have many problems with pollution and traffic that is getting worse. | The people who live in the city have many problems with pollution and traffic that is getting worse. | yes |
| He was late because he forget his keys and then he realize that he leave it on the table in the kitchen. | He was late because he forgot his keys and then realized that he left them on the table in the kitchen. | He was late because he forgot his keys and then he realized that he left them on the table in the kitchen. | no |
| In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he told that the company is planning to expand their services to other countries. | In the meeting yesterday, he said that the company is planning to expand their services in other countries. | no |
| Many students finds difficult to concentrates in online class because of distractions and lack of motivations. | Many students find it difficult to concentrate in online class because of distractions and lack of motivation. | Many students find it difficult to concentrate in online classes because of distractions and lack of motivation. | no |
| She don’t has enough experience, so they didn’t hired her for the job. | She doesn't have enough experience, so they didn't hire her for the job. | She doesn't have enough experience, so they didn't hire her for the job. | yes |
| The team were not perform well last game because they didn’t prepared properly. | The team did not perform well last game because they didn't prepare properly. | The team did not perform well last game because they weren't properly prepared. | no |
| I am agree with your opinion, but we needs more evidence before taking decision. | I agree with your opinion, but we need more evidence before taking a decision. | I agree with your opinion, but we need more evidence before making a decision. | no |
| He want to know what do means by artificial intelligences in modern technologies. | He wants to know what artificial intelligences in modern technologies mean. | He wants to know what is meant by artificial intelligence in modern technologies. | no |
| This kind of mistakes is happens when we not focus during the test. | This kind of mistake happens when we are not focused during the test. | This kind of mistake happens when we are not focused during the test. | yes |
| they is here today. | They are here today. | They are here today. | yes |
| she go to school. | She goes to school. | She goes to school. | yes |
| He readed the book. | He read the book. | He read the book. | yes |
| She can sings well. | She can sing well. | She can sing well. | yes |
| She go to school. | She goes to school. | She goes to school. | yes |

## Audio formats (public demo)

- m4a (AAC, file dari iPhone): HTTP 200, heard "She can sing well."
- mp4 terfragmentasi (AAC, MediaRecorder Safari): HTTP 200, heard "She can sing well."
- webm (Opus, Chrome/Firefox): HTTP 200, heard "She can sing well."
- ogg (Opus): HTTP 200, heard "She can sing well."
- flac: HTTP 200, heard "She can sing well."

## Unusual inputs (public demo)

- `Saya pergi ke sekolah setiap hari.` → `Saya pergi ke sekolah setiap hari.`
- `aku suka makan nasi goreng and I likes apples` → `Akan nasi goreng and I like apples.`
- `He don't like 🍎 apples.` → `He doesn't like apples.`
- `I bought 3.5 kg of rice at 7.30 a.m. yesterday.` → `I bought 3.5 kg of rice at 7.30 a.m. yesterday.`
- `He said: "she go home" (yesterday).` → `He said: "She goes home" (yesterday).`
- `asdf qwer zxcv` → `Asdf qwer zxcv.`
- `!!! ??? ...` → `!!! ??? ...`
- `12345` → `12345.`
- `um uh I I go go to to school` → `Um uh, I go to school.`
- `SHE GO TO SCHOOL` → `She goes to school.`
- `she dont like apple` → `She doesn't like an apple.`
- `The quick brown fox jumps over the lazy dog.` → `The quick brown fox jumps over the lazy dog.`
- `My name is Budi and I live in Surabaya.` → `My name is Budi and I live in Surabaya.`
- `Yesterday I go to the market with my father. My mother cook rice. We eats together.` → `Yesterday I went to the market with my father. My mother cooks rice. We eat together.`
- `Dr. Smith don't like it. Mr. Budi go home.` → `Dr. Smith don't like it. Mr. Budi goes home.`
- `A` → `A.`
