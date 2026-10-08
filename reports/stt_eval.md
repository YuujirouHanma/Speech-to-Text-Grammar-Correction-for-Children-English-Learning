# STT evaluation (15 recordings, device=cpu)

| Model | Verbatim prompt | WER (lower=better) | Verbatim rate (higher=better) | Real-time factor |
|---|---|---|---|---|
| base.en | no | 0.050 | 53% | 0.095 |
| base.en | yes | 0.039 | 67% | 0.092 |
| small.en | no | 0.044 | 53% | 0.266 |
| small.en | yes | 0.039 | 60% | 0.291 |

## base.en / prompt=no

| Reference (spoken) | Transcript | Word errors |
|---|---|---|
| Although she studied hardly for the exam but she didn’t passed and she was very disappointing about the results. | Although she studied hardly for the exam, but she didn't passed and she was very disappointing about the results. | 0 |
| He readed the book  | He read the book. | 1 |
| He want to know what do means by artificial intelligences in modern technologies. | He want to know what do means by artificial intelligences in modern technologies. | 0 |
| He was late because he forget his keys and then he realize that he leave it on the table in the kitchen. | He was late because he forget his keys and then he realized that he leave it on the table in the kitchen. | 1 |
| I am agree with your opinion, but we needs more evidence before taking decision. | I am agree with your opinion, but we need more evidence before taking decision. | 1 |
| In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | 0 |
| Many students finds difficult to concentrates in online class because of distractions and lack of motivations. | Many students find it difficult to concentrate in online class because of distractions and lack of motivations. | 3 |
| She don’t has enough experience, so they didn’t hired her for the job. | She don't has enough experience, so they didn't hired her for the job. | 0 |
| she go to school | She go to school. | 0 |
| The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | The peoples who lives in the city has many problem with pollution and traffic that is getting more worst. | 1 |
| The team were not perform well last game because they didn’t prepared properly. | The team were not perform well last game because they didn't prepared properly. | 0 |
| they is here today | They is here today. | 0 |
| This kind of mistakes is happens when we not focus during the test. | This kind of mistakes as happens when we not focus during the test. | 1 |
| She can sings well. | She can sing well. | 1 |
| She go to school. | She go to school. | 0 |

## base.en / prompt=yes

| Reference (spoken) | Transcript | Word errors |
|---|---|---|
| Although she studied hardly for the exam but she didn’t passed and she was very disappointing about the results. | Although she studied hardly for the exam, but she didn't passed and she was very disappointing about the results. | 0 |
| He readed the book  | He read the book. | 1 |
| He want to know what do means by artificial intelligences in modern technologies. | He want to know what do means by artificial intelligences in modern technologies. | 0 |
| He was late because he forget his keys and then he realize that he leave it on the table in the kitchen. | He was late because he forget his keys and then he realized that he leave it on the table in the kitchen. | 1 |
| I am agree with your opinion, but we needs more evidence before taking decision. | I am agree with your opinion, but we need more evidence before taking decision. | 1 |
| In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | 0 |
| Many students finds difficult to concentrates in online class because of distractions and lack of motivations. | Many students find it difficult to concentrate in online class because of distractions and lack of motivations. | 3 |
| She don’t has enough experience, so they didn’t hired her for the job. | She don't has enough experience, so they didn't hired her for the job. | 0 |
| she go to school | She go to school. | 0 |
| The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | The peoples who lives in the city has many problem with pollution and traffic that is getting more worst. | 1 |
| The team were not perform well last game because they didn’t prepared properly. | The team were not perform well last game because they didn't prepared properly. | 0 |
| they is here today | They is here today. | 0 |
| This kind of mistakes is happens when we not focus during the test. | This kind of mistakes is happens when we not focus during the test. | 0 |
| She can sings well. | She can sings well. | 0 |
| She go to school. | She go to school. | 0 |

## small.en / prompt=no

| Reference (spoken) | Transcript | Word errors |
|---|---|---|
| Although she studied hardly for the exam but she didn’t passed and she was very disappointing about the results. | although she studied hardly for the exam but she didn't pass and she was very disappointing about the results. | 1 |
| He readed the book  | He read the book. | 1 |
| He want to know what do means by artificial intelligences in modern technologies. | He want to know what do means by artificial intelligences in modern technologies. | 0 |
| He was late because he forget his keys and then he realize that he leave it on the table in the kitchen. | He was late because he forget his keys and then he realized that he leave it on the table in the kitchen. | 1 |
| I am agree with your opinion, but we needs more evidence before taking decision. | I am agree with your opinion, but we needs more evidence before taking decision. | 0 |
| In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | 0 |
| Many students finds difficult to concentrates in online class because of distractions and lack of motivations. | Many students find difficult to concentrate in online class because of distractions and lack of motivations. | 2 |
| She don’t has enough experience, so they didn’t hired her for the job. | She don't has enough experience, so they didn't hired her for the job. | 0 |
| she go to school | She go to school. | 0 |
| The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | The peoples who lives in the city has many problem with pollution and traffic that is getting more worst. | 1 |
| The team were not perform well last game because they didn’t prepared properly. | The team were not perform well last game because they didn't prepared properly. | 0 |
| they is here today | They is here today. | 0 |
| This kind of mistakes is happens when we not focus during the test. | This kind of mistakes as happens when we not focus during the test. | 1 |
| She can sings well. | She can sing well. | 1 |
| She go to school. | She go to school | 0 |

## small.en / prompt=yes

| Reference (spoken) | Transcript | Word errors |
|---|---|---|
| Although she studied hardly for the exam but she didn’t passed and she was very disappointing about the results. | Although she studied hardly for the exam but she didn't past and she was very disappointing about the results. | 1 |
| He readed the book  | He read the book. | 1 |
| He want to know what do means by artificial intelligences in modern technologies. | He want to know what do means by artificial intelligences in modern technologies. | 0 |
| He was late because he forget his keys and then he realize that he leave it on the table in the kitchen. | He was late because he forget his keys and then he realized that he leave it on the table in the kitchen. | 1 |
| I am agree with your opinion, but we needs more evidence before taking decision. | I am agree with your opinion, but we needs more evidence before taking decision. | 0 |
| In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | In the meeting yesterday, he tell that the company are planning to expanding their services in other countries. | 0 |
| Many students finds difficult to concentrates in online class because of distractions and lack of motivations. | Many students find difficult to concentrate in online class because of distractions and lack of motivations. | 2 |
| She don’t has enough experience, so they didn’t hired her for the job. | She don't has enough experience, so they didn't hired her for the job. | 0 |
| she go to school | She go to school. | 0 |
| The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | The peoples who lives in the city has many problem with pollution and traffics that is getting more worst. | 0 |
| The team were not perform well last game because they didn’t prepared properly. | The team were not perform well last game because they didn't prepared properly. | 0 |
| they is here today | They is here today. | 0 |
| This kind of mistakes is happens when we not focus during the test. | This kind of mistakes as happens when we not focus during the test. | 1 |
| She can sings well. | She can sing well. | 1 |
| She go to school. | She go to school. | 0 |
