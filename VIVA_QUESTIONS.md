# VIVA / INTERVIEW QUESTIONS — 45 Questions with Short Answers

Preparation for a technical viva or interview about the Customer Review Sentiment Analysis project.

**How to use this:** cover the answers, read a question, say your answer out loud, then check. Speaking it is the practice that counts — reading it is not.

Answers are kept **short on purpose** — the length you would actually speak. For the deeper explanation of any concept, see **[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)**.

At the end: a **[60-second](#60-second-project-explanation)** and a **[2-minute](#2-minute-technical-explanation)** spoken explanation of the project.

---

## Contents

- [A. NLP and the problem (Q1–6)](#a-nlp-and-the-problem)
- [B. Data (Q7–13)](#b-data)
- [C. Features and TF-IDF (Q14–22)](#c-features-and-tf-idf)
- [D. Models (Q23–29)](#d-models)
- [E. Evaluation (Q30–37)](#e-evaluation)
- [F. Overfitting and generalisation (Q38–40)](#f-overfitting-and-generalisation)
- [G. Engineering and deployment (Q41–45)](#g-engineering-and-deployment)
- [The hard questions you should expect](#the-hard-questions-you-should-expect)
- [60-second project explanation](#60-second-project-explanation)
- [2-minute technical explanation](#2-minute-technical-explanation)

---

## A. NLP and the problem

### Q1. What is NLP?

Natural Language Processing — the field of computer science that deals with human language like English or Urdu. The core difficulty is that computers do arithmetic while language is not arithmetic: a computer can compare `5 > 3` instantly but has no idea whether `"decent"` is closer to `"good"` or `"bad"`. NLP is the set of techniques for bridging that gap. It covers translation, summarisation, spam detection, chatbots and sentiment analysis.

### Q2. What is sentiment analysis?

Teaching a computer to decide whether a piece of writing is positive or negative. Also called opinion mining. A human reads *"the battery died after two days"* and instantly knows the customer is unhappy — sentiment analysis is getting software to reach the same conclusion. Mine is **binary**: Positive or Negative. Other versions are three-class with neutral, 1–5 star prediction, and aspect-based, which gives a separate sentiment per topic.

### Q3. What problem does your project solve?

A business gets far more written reviews than anyone can read, and the important ones — the complaints — are buried. My project takes review text and predicts Positive or Negative with a confidence score. That gives you scale (thousands tagged in seconds), prioritisation (route angry reviews to support first), trend tracking (satisfaction dropping month over month), and insight — because the model keeps one weight per word, you can read off which words unhappy customers actually use.

### Q4. Is this supervised or unsupervised learning? Why?

**Supervised.** Every review in my training data comes with a human-assigned label — 1 for positive, 0 for negative. The model learns by comparing its guess to the known right answer. Unsupervised learning has no labels; you would only be able to *cluster* reviews into groups, not name those groups "positive" and "negative".

### Q5. Is this classification or regression?

**Classification** — the output is a category, Positive or Negative. Regression predicts a continuous number. If I were predicting a star rating from 1.0 to 5.0, that would be regression.

### Q6. Why didn't you use deep learning?

Three reasons. First, it was the point of the exercise — to demonstrate that I understand the classical fundamentals rather than importing a solution. Second, **interpretability**: Logistic Regression has one weight per word, so my app can show exactly which words drove any prediction; a transformer gives you a number and no explanation. Third, **the data doesn't justify it** — with 2,982 short sentences a fine-tuned transformer would likely overfit, and it would need a GPU and far more time.

And I know what I gave up: a fine-tuned DistilBERT would handle word order and sarcasm much better. But having a measured, interpretable baseline is exactly what tells you whether that complexity is worth paying for.

---

## B. Data

### Q7. What dataset did you use?

The **Sentiment Labelled Sentences** dataset from the UCI Machine Learning Repository — 3,000 real review sentences, 1,000 each from Amazon (products), Yelp (restaurants) and IMDb (movies), each labelled 1 for positive or 0 for negative. It's balanced by design: 500 positive and 500 negative per site. It's from Kotzias et al., KDD 2015. After cleaning I had **2,982 usable rows**.

I chose it because it's small, genuinely public, genuinely human-labelled, and made of real customer review text — so nothing in the project rests on invented data.

### Q8. Why split the dataset into train and test?

Because **a model can memorise.** If I train on all the data and test on the same data, I'm asking "do you remember what I just told you?", not "have you learned anything useful?" Real users will send reviews the model has never seen, so the only honest measurement is on reviews the model has never seen.

The analogy: revising a past paper is training. If the exam is literally that same past paper, a high score proves nothing.

### Q9. Why 80/20, and why stratified?

**80/20** is a trade-off — more training data gives a better model, more test data gives a more trustworthy score. With 2,982 rows that gives 2,385 training and 597 test reviews.

**Stratified** means both halves keep the same positive/negative ratio, so my test set is a fair miniature of the whole dataset — 299 negative and 298 positive. Without it, random chance could hand me a test set that is 60% positive, and my score would be measuring luck. It matters even more on imbalanced data: with a 95/5 split a random test set might contain almost no minority examples at all, so you couldn't measure the thing you care most about.

### Q10. Why `random_state=42`?

The split is random, and `random_state` fixes the random number generator so the **same split happens every time**. That makes my results reproducible — anyone who runs the training script gets 82.08%, and my separate evaluation script can rebuild the identical test set later to verify the saved model. 42 is just a convention.

### Q11. What cleaning did you do?

Two layers. **Row-level:** normalise labels to 0/1, drop rows with missing or blank text, and drop duplicate reviews. That removed 18 of the 3,000 rows. **Text-level:** lowercase, strip URLs, strip HTML tags, remove punctuation and symbols, collapse whitespace.

Anything with an unrecognisable label — like `"neutral"` — is **dropped, not guessed**, because a wrong label actively teaches the model something false.

### Q12. Why drop duplicate reviews?

Not for tidiness — to prevent leakage. If the same sentence appears twice and the split puts one copy in train and the other in test, the model has **already seen the test answer**. My test score goes up without the model getting any better.

### Q13. Did you remove stopwords? Why not?

**No, deliberately.** Removing stopwords is standard NLP advice and it's **wrong for sentiment analysis**, because the most dangerous stopword is `"not"`. Remove it from *"not good"* and you get *"good"* — the exact opposite meaning.

And my model proves it matters: `"not"` is the **strongest negative feature it learned, at −9.33**. Deleting it would have thrown away the single most useful word in the vocabulary.

I do provide an opt-in stopword helper, but its list deliberately excludes `not`, `no`, `never`, `but`, `very` and `too`.

---

## C. Features and TF-IDF

### Q14. What is a feature?

One measurable input the model learns a weight for. In my project **one feature = one term in the vocabulary** — `"great"` is a feature, and so is the two-word feature `"not good"`. I have 20,000 of them, and Logistic Regression learned one weight per feature.

### Q15. Why can't you feed raw text into Logistic Regression?

Because Logistic Regression is arithmetic — it multiplies each input by a weight and adds them up. **You cannot multiply the word `"terrible"` by 0.7.** The model needs numbers, and it needs a *fixed number* of them in a fixed order, so that weight number 4,231 always means the same thing. Reviews are different lengths and made of words, so text has to be converted into a fixed-length vector of numbers first. That conversion is feature extraction, and I use TF-IDF for it.

### Q16. What is TF-IDF?

**Term Frequency × Inverse Document Frequency.** Two ideas multiplied:

- **TF** — how often the word appears in *this* review. Used a lot here, so probably important here.
- **IDF** — how rare the word is across *all* reviews. `"the"` is everywhere so it gets a tiny weight; `"refund"` is rare so it gets a big one.

The multiplication is the clever bit: a word scores high **only if it's frequent here and rare overall** — which is a good working definition of "informative".

### Q17. Give me the TF-IDF formula.

```
TF-IDF(word, review) = TF × IDF

TF  = how many times the word appears in that review

            ⎛  1 + total reviews            ⎞
IDF = ln ⎜ ───────────────────────────── ⎟ + 1
            ⎝  1 + reviews containing word  ⎠
```

Then each row is L2-normalised — divided by its own length — so a long review doesn't outweigh a short one just for having more words. The `+1`s are smoothing that avoids dividing by zero, and the logarithm stops a word that appears in 1 review out of a million from getting a weight of a million.

### Q18. What is term frequency?

How many times a word appears in one document. In *"the food was good, really good"*, `"good"` has TF = 2 and everything else has TF = 1. The intuition is that a repeated word is probably central to that review. **Its weakness alone:** the highest TF in any English text belongs to `"the"` and `"is"`, which tell you nothing — which is exactly the gap IDF fills.

### Q19. What is inverse document frequency?

A score that is **high for rare words and low for common words**. Document frequency is how many documents contain the word; *inverse* means we flip it.

In my trained model: `"the"` has an IDF of **1.805**, `"food"` has **4.223**. Two things matter about IDF — it's computed once during `fit` from the training documents only and then frozen, and it's a property of the whole corpus rather than any one review.

The neat consequence: **IDF gives you automatic, data-driven stopword removal.** I never wrote a list of boring words; the maths noticed `"the"` is everywhere and de-weighted it by itself.

### Q20. Why TF-IDF instead of simple word counts?

With plain counts, in the sentence *"the food was not good"*, every word scores 1 — so `"the"` looks as important as `"good"`. That's obviously wrong.

With TF-IDF, using my actual vectorizer: `"the"` gets 0.1336 and `"food"` gets 0.3127 — **2.3× the weight**, purely because the maths noticed `"the"` is everywhere. You also get length normalisation for free.

**The honest caveat:** TF-IDF isn't always better. Multinomial Naive Bayes was designed for raw counts and sometimes prefers them. And TF-IDF doesn't fix the real weaknesses of bag-of-words — it still ignores word order and still has no idea `"great"` and `"excellent"` are near-synonyms.

### Q21. What is an n-gram? Which did you use and why?

An n-gram is a sequence of n consecutive words. For *"not good"*: the unigrams are `not` and `good`; the bigram is `not good`.

I used `ngram_range=(1, 2)` — **unigrams and bigrams**. The reason is negation. Bag-of-words throws away word order, so with unigrams alone *"not good"* becomes two independent features, and `good` is one of my strongest *positive* words — the negation is invisible. Bigrams make `"not good"` its own feature that can carry its own weight.

**And it demonstrably worked.** My model learned:

| Feature | Weight |
| --- | --- |
| `good` | **+8.14** |
| `not good` | **−3.45** |

It learned that the phrase means something different from the word. Measured, bigrams beat unigrams-only by 0.2–0.4 accuracy points.

### Q22. Why not trigrams?

Cost grows fast and returns collapse. Bigrams already took my vocabulary to 21,269 features; trigrams would multiply it again. And longer n-grams are rarer, so most would appear once or twice — not enough examples to learn a reliable weight, which is overfitting waiting to happen. Sentiment negation is mostly a two-word pattern (`not good`, `never again`, `too expensive`), so bigrams capture most of the available benefit. `(1,2)` is the standard sweet spot.

---

## D. Models

### Q23. What is Logistic Regression?

A model that predicts a probability between 0 and 1, used for classification despite the name. Three steps:

1. **Give every feature a weight** — that's what training learns. Mine learned `great = +10.49`, `bad = −7.43`.
2. **Add everything up** — each feature's value times its weight, plus an intercept. That total can be any number.
3. **Squash it into 0–1 with the sigmoid function**, `1/(1+e^−z)`. Above 0.5 → Positive.

### Q24. Why did you choose Logistic Regression?

Four reasons.

**Interpretability first** — one weight per word means I can read the model and see it learned `great = +10.49` and `not = −9.33`, and I can explain any single prediction by listing which words contributed what. My app does exactly that.

**It suits the data shape** — 20,000 features and 2,385 examples. Linear models handle high-dimensional sparse data comfortably.

**Text really is roughly linear** — sentiment behaves additively, so a linear model matches the shape of the problem.

**It's the standard baseline** — TF-IDF plus Logistic Regression is what a practitioner tries first on text. If something fancier can't beat it, the complexity isn't worth paying for.

And I verified it rather than assuming: I compared it against Multinomial Naive Bayes and it won on the test set, 82.08% to 81.07%.

### Q25. What is Naive Bayes, and why is it "naive"?

A probabilistic classifier built on Bayes' theorem. It asks: *which class makes this combination of words most likely?* Training is essentially just counting how often each word appears in each class — so it's extremely fast.

It's **"naive"** because it assumes every word is independent of every other word given the class. That's plainly false — `"not"` and `"good"` appearing together is no coincidence.

**But it still works well,** and the reason is subtle: the assumption makes the *probability estimates* badly wrong, but classification only needs the *ranking* of the classes to be right. You can be quite wrong about the numbers and still pick the right winner.

### Q26. Why compare two models?

Because a single number means nothing alone — "82% accuracy", is that good? You can't tell without a comparison. A baseline tells you whether extra complexity earns its keep, and no model is universally best, so trying more than one is the only way to know which suits your data.

In my case the comparison also taught me something useful: the models differ by **1.0 percentage point** while my cross-validation varies by **±1.2 points**. So the honest conclusion is *"Logistic Regression is slightly ahead and more interpretable"* — not *"Logistic Regression is better"*. Recognising that a gap sits inside the noise matters.

### Q27. What is `C` in Logistic Regression, and how did you choose it?

`C` is the **inverse of the regularisation strength.** Regularisation penalises large weights to stop the model memorising. Small `C` = strong penalty = weights squashed towards zero = simpler model. Large `C` = weights free to grow = more flexible.

I chose `C=10` by **5-fold cross-validation on the training set only** — never on the test set:

| C | 1 | 2 | 5 | **10** | 20 | 50 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CV accuracy | 0.810 | 0.821 | 0.831 | **0.839** | 0.842 | 0.839 | 0.840 |

The curve flattens from 10 onwards and everything above is inside one standard deviation. So I took the **smallest `C` on the plateau** — same score, more regularisation, simpler model.

*(If they push: "why not 20, it scored highest?" — because the difference is inside the noise, and between two statistically equal models you take the simpler one.)*

### Q28. What's the difference between a parameter and a hyperparameter?

A **parameter** is learned by the model during training — my 20,000 weights. A **hyperparameter** is set by me *before* training — `C`, `min_df`, `ngram_range`. You tune hyperparameters with cross-validation on the training set, never on the test set.

### Q29. What is cross-validation, and why did you need it?

5-fold cross-validation splits the **training** data into 5 parts, then trains 5 times, each time holding out a different part to validate on, and averages the 5 scores. Every row gets used for both training and validating, just never at the same time.

I needed it for two reasons: it gives a **more stable estimate** than one single split, plus a standard deviation that tells me how much the number wobbles; and crucially it let me **tune hyperparameters without ever touching the test set**. My result was 0.8377 ± 0.0119.

---

## E. Evaluation

### Q30. What accuracy did you get?

**82.08% on the test set** — 490 of 597 unseen reviews correct. Cross-validation on the training set said 0.8377 ± 0.0119, so the two agree, which is reassuring. Naive Bayes got 81.07%. My F1-macro is 0.8208 and ROC-AUC is 0.8952.

### Q31. What is accuracy?

The fraction of predictions that were correct: `(TP + TN) / everything`. Mine is `(246 + 244) / 597 = 0.8208`. Its strength is that everyone understands it instantly.

### Q32. Why is accuracy sometimes misleading?

**Class imbalance.** Imagine 1,000 reviews where 950 are positive. A one-line model that always answers "Positive" scores **95% accuracy** — and it's completely useless, because it never once identifies an unhappy customer, which is the whole business reason for building it. Its recall on the negative class is **zero**.

Accuracy hides that because it lumps both classes together. Precision, recall, F1 and the confusion matrix expose it immediately. I have that exact scenario as a unit test.

**For my project specifically:** my data is ~50/50 balanced, so accuracy *is* a fair headline number here — the coin-flip baseline is 50%, so 82% is real learning. But I still report the full set, because on the next dataset it might not be.

### Q33. What is precision?

*"Of everything I labelled positive, how much really was positive?"* — `TP / (TP + FP)`.

Mine is **0.8215**: I called 297 reviews positive and 244 really were. **Precision is about false alarms.** It matters most when acting on a wrong positive is expensive — a spam filter needs high precision, because sending a real job offer to the spam folder is far worse than letting one spam email through.

### Q34. What is recall?

*"Of everything that really was positive, how much did I find?"* — `TP / (TP + FN)`.

Mine is **0.8188**: there were 298 genuinely positive reviews and I found 244. **Recall is about misses.** It matters most when missing a case is expensive — cancer screening needs high recall, because a false alarm means one more test but a missed tumour can be fatal.

### Q35. What's the trade-off between them?

They pull against each other, and the dial is the **decision threshold** — I use 0.5. Raise it to 0.9 and you only call the very confident cases positive: precision up, recall down. Lower it to 0.1 and you call almost everything positive: recall up, precision down.

Which to favour is a **business decision, not a maths one.** For flagging angry customers I'd favour recall — better to over-flag than let a furious customer go unanswered.

In my results precision (0.8215) and recall (0.8188) are almost identical, which is exactly what you'd want on balanced data at a 0.5 threshold: the model isn't biased towards either sentiment.

### Q36. What is F1-score, and why the harmonic mean?

F1 combines precision and recall: `2PR / (P + R)`. Mine is **0.8202**.

It's the **harmonic** mean because that punishes imbalance. Take precision 1.0 and recall 0.02 — the "only predict when 99.9% sure" model. A simple average gives 0.51, which looks acceptable. F1 gives **0.039**, which is correctly brutal. The harmonic mean sits close to the *smaller* of the two numbers, so **you can't get a good F1 by maximising one metric and ignoring the other** — which is exactly the behaviour you want to prevent.

### Q37. What is a confusion matrix? Show me yours.

A table showing exactly which mistakes the model made. Every other metric can be computed from it. Mine, on 597 test reviews:

|  | Predicted Negative | Predicted Positive |
| --- | --- | --- |
| **Actual Negative** | **246** ✅ TN | **53** ❌ FP |
| **Actual Positive** | **54** ❌ FN | **244** ✅ TP |

The naming trick: the second word is what you **predicted**, the first is whether you were **right**. So "False Positive" = predicted Positive, and wrong.

**What mine tells me:** 107 errors total, split almost evenly — 53 versus 54. So the model isn't systematically over-cheerful or over-harsh. If I'd seen 10 FP and 97 FN I'd know it was biased towards "negative" and could fix it by moving the threshold.

For my use case a **false positive is the worse error** — that's an unhappy customer I never saw. So in production I'd consider lowering the threshold below 0.5 to catch more negatives, accepting more false alarms.

---

## F. Overfitting and generalisation

### Q38. What is overfitting? Is your model overfitting?

Overfitting is when a model memorises its training data instead of learning general patterns — brilliant on data it's seen, poor on anything new. Like a student who memorised last year's exam answers.

**Yes, mine is overfitting, and I'll be straight about it.** Train accuracy is **1.0000**; test accuracy is **0.8208**. That's an 18-point gap — it classifies its training data perfectly.

Three things about that:

1. **It's near-inevitable at this shape.** With 20,000 features and 2,385 examples there are more dimensions than data points, so a linear model can almost always separate the training set exactly. Text classification nearly always looks like this.
2. **The honest estimates agree with each other.** Cross-validation said 0.8377, the untouched test set said 0.8208. If the test score had collapsed to 0.60, the gap would be a real problem — the model *did* learn something that generalises.
3. **I tried reducing it and it made things worse.** At `C=1` the gap narrows but real accuracy drops from 0.839 to 0.810. A smaller gap with a worse model isn't an improvement.

So the gap is a symptom of a **small dataset**, not bad modelling. More data is the fix, not more regularisation.

**And here's concrete evidence of it:** my model gives the word `"and"` a weight of **+4.53**. `"and"` has no sentiment. It scored highly because in this corpus positive sentences happen to string clauses together — *"good value and fast delivery"*. That's the model learning a quirk of 2,385 sentences rather than learning English.

### Q39. What is underfitting? Did you see it?

The opposite — the model is too simple to capture the pattern, so it does badly on training data **and** test data. Like a student who didn't revise at all. The giveaway is a **low training score**.

**Yes, I saw it.** At `C=1` my cross-validation accuracy was only 0.810 versus 0.839 at `C=10`. The regularisation penalty was so strong it crushed the weights and the model couldn't use the evidence available to it. Raising `C` fixed it — the textbook cure.

*Bonus, if asked how they relate:* that's the **bias–variance trade-off**. Bias is error from wrong assumptions (underfitting), variance is error from being too sensitive to the particular training data (overfitting). You can't minimise both; you find the sweet spot. That's literally what my `C` grid search was doing — walking from high-bias at `C=1` to high-variance at `C=100` and picking the best point in between.

### Q40. What is data leakage? How did you avoid it?

Leakage is when information from the test set sneaks into training. The result is a score that **looks great and is a lie** — and it's dangerous because nothing crashes, your numbers just quietly become wrong.

**The specific leak in NLP projects** is fitting the vectorizer before splitting:

```python
X = vectorizer.fit_transform(all_reviews)    # ❌ leaks
X_train, X_test = train_test_split(X, ...)
```

`fit` does two things: builds the vocabulary and computes the IDF weights. If it sees everything, then test-set words get vocabulary columns and every IDF value is calculated using test-set document counts. The training features were shaped by test data, so the score comes out optimistically high.

**What I do instead:**

```python
X_train_text, X_test_text, ... = train_test_split(...)   # split FIRST
X_train = vectorizer.fit_transform(X_train_text)         # fit on train
X_test  = vectorizer.transform(X_test_text)              # transform only
```

**`fit_transform` on train, `transform` on test, never `fit` on test.**

I also guard two other kinds: **duplicate leakage** — I drop duplicate reviews so the same sentence can't be in both halves; and **test-set tuning** — I chose `C` and `min_df` with cross-validation *inside* the training set. The test set was used exactly once, at the end.

---

## G. Engineering and deployment

### Q41. How did you save the model? Why save the vectorizer too?

`joblib.dump()` to two files — the fitted Logistic Regression and the fitted TF-IDF vectorizer. joblib is preferred over pickle for scikit-learn because it handles large NumPy arrays efficiently.

**Saving the vectorizer is essential, and it's the mistake beginners make.** The model is just a list of 20,000 weights. Weight number 4,231 might mean `"not good"` — but the model doesn't know that. **The word-to-column mapping lives inside the vectorizer.** If I saved only the model and rebuilt the vectorizer later, I'd get a different vocabulary in a different order, weight 4,231 would be applied to some unrelated word, and every prediction would be silently wrong. Not crashed — *wrong*, which is worse.

They're a matched pair, and I have a test asserting `model.coef_.shape[1] == len(vectorizer.vocabulary_)`.

The same logic means my `clean_text()` function must run identically at training and prediction time — and I have a test asserting `"THIS IS GREAT!!!"` and `"this is great"` give identical answers.

### Q42. How does Streamlit work?

It turns a Python script into a web app — no HTML, CSS or JavaScript. The key thing to understand is the execution model: **Streamlit re-runs the entire script top to bottom on every interaction** — every click, every keystroke.

That's what makes the code simple, but it means **anything expensive must be cached** or it happens on every click. I use `@st.cache_resource` for the model (live objects, shared across users) and `@st.cache_data` for the metrics JSON (plain data). Without that, the app would reload both `.joblib` files from disk every time someone typed a character.

**My app never trains** — it only loads the saved artifacts. And one detail I'm happy with: the performance figures in the sidebar are read from `reports/metrics.json` rather than typed into the app, so the UI can't drift into advertising a number the model never achieved.

### Q43. How would you deploy this model?

In stages, by maturity.

**Simplest:** push it to Streamlit Community Cloud — connect the repo and it's live.

**Properly:** wrap `predict.py` in a **FastAPI** endpoint that accepts JSON and returns the label and probability, containerise it with **Docker** so the environment is reproducible, and put it behind a load balancer. The Streamlit app then becomes just one client of that API.

**Production concerns I'd need to handle:** log every prediction so I can monitor **model drift** — language changes as new products and slang appear, so a deployed model slowly gets worse and needs periodic retraining. I'd also version the model artifacts, add a health-check endpoint, and set up a retraining pipeline. Since the model is only kilobytes and prediction is a single sparse matrix multiply, it'll serve thousands of requests per second on modest hardware.

### Q44. How did you test this project?

**86 pytest tests** across four files. The ones I'd highlight:

- **Preprocessing** — cleaning handles case, URLs, HTML, punctuation, digits, `None`/`NaN` input, and is **idempotent** (cleaning clean text changes nothing), which matters because training and prediction both call it.
- **Features** — that bigrams are actually created, that the matrix is sparse, that rows are L2-normalised, and that **a rare word gets a higher IDF than a common one** — that's the IDF concept as an assertion.
- **Evaluation** — metrics checked against a **hand-worked 10-row example** I can verify with a pen, plus the always-predict-one-class trap as a test.
- **Prediction** — that the model's weight count matches the vectorizer's vocabulary, that a joblib round-trip gives identical predictions, that batch and single prediction never disagree, and that **row order is preserved when blank rows are mixed into a batch**.

The prediction tests skip themselves with a clear message if the model hasn't been trained, so a fresh clone gives a useful result rather than a confusing failure.

### Q45. What does your project structure look like, and why?

`src/` holds one module per pipeline stage — `preprocessing`, `features`, `train`, `evaluate`, `predict` — plus a `config.py` that holds every path and setting in one place so nothing is hardcoded twice. `app.py` is the Streamlit UI, `tests/` mirrors `src/`, `data/` and `models/` hold artifacts and are gitignored because they're rebuildable with one command each.

The principle is **separation of concerns**: `train.py` knows how to train and nothing about the UI; `app.py` knows how to display and nothing about training. That's why the same `predict.py` serves both the command line and the web app without duplication.

---

## The hard questions you should expect

These are the follow-ups that separate a memorised answer from an understood one.

### "82% isn't very good. Why so low?"

It's honest, and I know exactly where the ceiling is. Three things cap it:

**The data.** 2,982 single sentences averaging 12 words. Published results on this dataset sit in a similar range for classical models — it's a small, deliberately mixed-domain corpus.

**The domains are mixed.** Movies, restaurants and electronics together. `"long"` is good for a battery and bad for a queue; one bag-of-words model can't hold both meanings.

**The method has a hard limit.** Bag-of-words can't see word order beyond bigrams, can't detect sarcasm, and doesn't know `"great"` and `"excellent"` are synonyms.

**What I'd do:** more and longer data first — that's the highest-value change by far, and the same code on 50,000 reviews would likely reach the high 80s. Then error analysis on my 107 mistakes. A transformer after that, if the accuracy gain justified losing interpretability.

The number I'd defend is not 82% — it's that **82% is measured honestly**, on data the model never saw, with no leakage, and cross-validation that agrees with it.

### "How would you handle sarcasm?"

Honestly: **my model can't, and that's inherent rather than a bug.** *"Great, it broke on day one"* contains `great`, my strongest positive feature at +10.49. A bag-of-words model has no mechanism for tone.

What would actually help, in order:

1. **A context-aware model** — a transformer reads words in relation to each other, so it can pick up the mismatch between a positive word and a negative event. This is the real answer.
2. **Sarcasm-labelled training data** — you can't learn a pattern you've never been shown, and general review corpora contain very little marked sarcasm.
3. **Extra signals** — sarcasm often comes with punctuation and casing cues (`"Great!!!"`, ALL CAPS, scare quotes). Those could be hand-built features, though they're weak on their own.
4. **Detect the mismatch** — flag reviews where strong positive words sit next to negative-outcome words (`broke`, `refund`, `returned`) as low-confidence for human review. A pragmatic partial fix.

The honest engineering answer: **rather than pretend to solve it, I'd surface uncertainty.** My app already reports confidence and how many words it recognised, so borderline cases can be routed to a human.

### "How would you add a neutral class?"

The code change is small; **the data change is the real work.**

**Code:** Logistic Regression handles multi-class natively — scikit-learn uses a one-vs-rest or multinomial scheme automatically. `predict_proba` returns three columns instead of two, and `LABEL_NAMES` gains a third entry. My metrics already use macro averaging, which handles three classes as-is. Maybe an hour of work.

**Data:** I need reviews *labelled* neutral, and my dataset has none — the authors deliberately excluded neutral sentences. So I'd need a differently-labelled dataset, or to label data myself.

**And neutral is genuinely harder**, which is worth saying. Positive and negative have distinctive vocabulary; neutral is defined by the *absence* of sentiment, so it has no characteristic words for TF-IDF to grab onto. Expect accuracy to drop, and expect most errors to be neutral-versus-something confusions.

**A cheaper interim option:** keep the binary model and treat low-confidence predictions as neutral — if the probability is between 0.4 and 0.6, report "unclear" instead of forcing a choice. That's not the same as a real neutral class, but it's honest and it takes ten lines.

### "How would you handle multilingual reviews?"

Four options, increasing in cost:

1. **Detect and route.** Use a language detector, then send each review to a model trained for that language. Clean, but you need labelled data per language.
2. **Translate, then classify.** Machine-translate everything to English and use my existing model. Cheap to build, but translation loses nuance — and sentiment lives in nuance.
3. **Train per language.** Best accuracy, most effort — you need a labelled corpus for each.
4. **Use a multilingual model.** Something like multilingual BERT or XLM-R shares meaning across languages, so it can even generalise to a language it saw little labelled data for. This is the right answer for real scale.

**Why my current approach doesn't extend:** TF-IDF features are literally language-specific strings. `"bueno"` is simply not in my vocabulary, so a Spanish review would contribute almost nothing and the prediction would be close to a coin flip. My app would at least *show* this — it reports how many words it recognised, so a Spanish review would visibly come back with near-zero recognised words.

One thing I'd get right cheaply: my `clean_text` uses `strip_accents="unicode"` and a regex that keeps `a-z0-9` only, which would **destroy non-Latin scripts entirely**. That regex would have to change first.

### "Your model gives `and` a weight of +4.53. Isn't that broken?"

It's a real flaw and I'd rather point it out than have it found. `"and"` carries no sentiment. It got a high weight because in this particular corpus positive sentences tend to string clauses together — *"good value and fast delivery"* — so the correlation is genuinely there in the training data.

That's the textbook illustration of **learning a corpus quirk instead of learning language**, and it's the clearest single piece of evidence that my model is overfitting a small dataset. More data would wash it out, because the correlation is an accident of 2,385 sentences rather than a fact about English.

It's also a good argument for *looking* at your model's weights rather than only its accuracy. I'd never have found this from the accuracy number alone.

### "Why is `min_df=1`? Everyone uses 2 or more."

Because I measured it, and the standard advice assumes a bigger corpus than mine.

The reasoning behind `min_df=2` is that a word appearing in only one document is probably a typo or a one-off name, so it can't generalise. That's correct **on a large corpus.** My training set is 2,385 short sentences, where a once-seen word is quite likely a real sentiment word like `"flawless"` or `"refund"` — there just isn't enough text for it to show up twice.

The cross-validation, on the training set only:

| `min_df` | 1 | 2 | 3 |
| --- | --- | --- | --- |
| CV accuracy | **0.839** | 0.829 | 0.819 |

Dropping those words cost a full point of accuracy. **Context beats convention, and measurement beats both** — and on a 500,000-review dataset I'd expect `min_df=2` to win, so I'd re-run the check rather than carry my answer over.

### "Walk me through what happens when a user types a review."

1. Streamlit reruns `app.py`; the button returns `True`.
2. `predict_sentiment(text)` is called.
3. `load_artifacts()` returns the cached model and vectorizer — no disk read after the first call.
4. `clean_text()` lowercases, strips URLs, HTML, punctuation, and collapses whitespace. **Exactly the same function used in training** — this is what keeps the features aligned.
5. `vectorizer.transform([cleaned])` produces one sparse row, 1 × 20,000. Only ~20 cells are non-zero. Unknown words are silently ignored.
6. `model.predict_proba(row)` multiplies each non-zero TF-IDF value by its learned weight, sums them with the intercept, and pushes the total through the sigmoid.
7. Above 0.5 → Positive. The probability of the winning class becomes the confidence.
8. The app shows the verdict, confidence, both probabilities, a plain-English note, and the top contributing terms — each one's TF-IDF value times its weight, which **is** the arithmetic from step 6, not an approximation of it.

If cleaning left nothing, or no word was recognised, it says so instead of guessing.

### "What was the hardest part / what did you learn?"

**The most valuable lesson was resisting the urge to guess at parameters.** My first run used the textbook defaults — `min_df=2`, `C=1.0`, `sublinear_tf=True` — and got **79.06%**, with Naive Bayes actually *beating* Logistic Regression. It would have been easy to write that up and move on.

Instead I cross-validated the choices on the training set: `min_df`, `C`, the n-gram range, and `sublinear_tf`. Three of the four textbook defaults turned out to be wrong **for this dataset**, and fixing them took me to **82.08%** — a 3-point gain with no change to the model or the data, and it flipped which model won.

What I took from it: **a default is a starting hypothesis, not an answer**, and the discipline of tuning on cross-validation rather than on the test set is what makes the final number trustworthy. It's also where the habit of writing the *evidence* for each parameter into the code comments came from.

---

# 60-SECOND PROJECT EXPLANATION

> *Use this when someone says "tell me about your project". Aim for calm and structured, not rushed.*

"I built a sentiment analysis system that reads a customer review and predicts whether it's positive or negative.

The problem it solves is scale — a business gets thousands of reviews and nobody can read them all, so complaints get buried.

I used the UCI Sentiment Labelled Sentences dataset: 3,000 real review sentences from Amazon, Yelp and IMDb, human-labelled positive or negative. After cleaning I had 2,982 usable rows, balanced roughly fifty-fifty.

The pipeline is classical machine learning end to end. I clean the text, split 80/20 with stratification, convert the text to numbers using TF-IDF with unigrams and bigrams, and train a Logistic Regression model. I also trained a Multinomial Naive Bayes as a baseline to compare against.

On 597 reviews the model had never seen, it got **82.1% accuracy**, with precision and recall both around 0.82 and an ROC-AUC of 0.90. Logistic Regression beat Naive Bayes by about a point.

Then I saved the model and the vectorizer with joblib and built a Streamlit web app on top. You type a review, it gives you the sentiment, a confidence score, and — because Logistic Regression keeps one weight per word — a list of exactly which words drove the prediction. It also does batch CSV upload.

The two things I'd point to: every parameter choice was made by cross-validating on the training set rather than guessing, and the model can explain any individual prediction. There are 86 tests, and the honest limitations are documented — no neutral class, and it can't detect sarcasm."

**(~60 seconds at a normal speaking pace.)**

---

# 2-MINUTE TECHNICAL EXPLANATION

> *Use this when the interviewer is technical and says "walk me through it".*

"**The problem.** Binary text classification — given a customer review, predict positive or negative. Supervised learning, because every training review comes with a human label.

**The data.** The UCI Sentiment Labelled Sentences dataset — 3,000 real review sentences, a thousand each from Amazon, Yelp and IMDb, labelled 1 or 0. It's balanced by design, 500 positive and 500 negative per source. My loader normalises the labels, drops missing text, and drops duplicate reviews — that last one matters, because a duplicate sitting in both halves of the split would leak the test answer. That left 2,982 usable rows out of 3,000.

**Preprocessing.** Deliberately light: lowercase, strip URLs and HTML, remove punctuation, collapse whitespace. I specifically **do not remove stopwords**, because the most dangerous stopword in sentiment analysis is 'not' — take it out of 'not good' and you get 'good', the opposite meaning. My trained model justifies that: 'not' is its strongest negative feature, at −9.33.

**The split.** 80/20, stratified so both halves keep the same class balance, with a fixed random state for reproducibility. 2,385 training, 597 test. I split **before** doing anything else — that's the key discipline.

**Features.** TF-IDF — term frequency times inverse document frequency. TF is how often a word appears in this review; IDF is how rare it is across all reviews. Multiplying them means a word scores high only when it's frequent here and rare overall, which is what makes it informative. I use unigrams **and bigrams**, because bag-of-words throws away word order and bigrams are the cheap way to catch negation. That worked: the model learned 'good' at +8.14 and the bigram 'not good' at −3.45 — it learned the phrase means something different from the word.

The output is a 2,385 by 20,000 sparse matrix, only 0.1% non-zero. And critically I **fit the vectorizer on the training text only** and just transform the test text. Fitting on everything would compute the IDF weights using test data — that's data leakage, and it silently inflates your score.

**The model.** Logistic Regression with `C=10`, compared against Multinomial Naive Bayes. I chose `C` by cross-validating a seven-value grid on the training set — 1 through 100. It plateaus at 10, and since everything above sits inside one standard deviation, I took the smallest C on the plateau: same accuracy, more regularisation, simpler model. I chose `min_df=1` the same way — it beat `min_df=2` by a full point, because on 2,385 short sentences a once-seen word is often a real sentiment word rather than a typo.

**Results.** 82.08% test accuracy — 490 of 597 correct. Precision 0.8215, recall 0.8188, F1-macro 0.8208, ROC-AUC 0.8952. The confusion matrix shows 107 errors split almost evenly, 53 false positives and 54 false negatives, so it isn't biased towards either sentiment. Naive Bayes got 81.07%, which is close enough that I'd call Logistic Regression *slightly* ahead and more interpretable, rather than better.

**I'll flag the overfitting myself:** training accuracy is 1.0 against 0.82 on test. With 20,000 features and 2,385 examples a linear model can separate the training set exactly — that's normal for text. What reassures me is that cross-validation said 0.8377 and the untouched test set said 0.8208; the two honest estimates agree. And I tried more regularisation — it narrowed the gap but made the model genuinely worse.

**Serving.** I save the model *and* the vectorizer with joblib. Both are essential — the model is just 20,000 weights, and the mapping from words to column numbers lives in the vectorizer. Save only the model and your weights line up against the wrong words, silently. The Streamlit app loads both and never retrains; there's a test asserting the weight count matches the vocabulary size.

**Limitations I'd name up front:** no neutral class, since the training data has none; sarcasm fails completely, because 'great, it broke on day one' contains my strongest positive word; and the model gives 'and' a weight of +4.53, which is real evidence it learned a quirk of this corpus rather than learning English.

**What I'd do next:** more and longer data, which is the actual ceiling here — then a neutral class, then proper error analysis on those 107 mistakes. A transformer would beat this and handle word order and sarcasm far better, but I'd lose the ability to explain every prediction by reading a weight, and having a measured baseline is exactly what tells you whether that trade is worth making."

**(~2 minutes at a normal speaking pace.)**

---

## Final preparation checklist

Before the viva, make sure you can do all of these **without notes**:

- [ ] Draw the pipeline on paper, from dataset to Streamlit app
- [ ] State the dataset, its size, and where it came from
- [ ] Explain TF-IDF in one sentence, then write the formula
- [ ] Explain why you cannot feed raw text to Logistic Regression
- [ ] Explain the sigmoid and how a probability becomes a label
- [ ] Give the four confusion-matrix cells and derive precision and recall from them
- [ ] Explain data leakage and the `fit_transform` / `transform` rule
- [ ] State your accuracy, F1, and ROC-AUC from memory
- [ ] **Admit the overfitting before you're asked, and explain why you accepted it**
- [ ] Name three limitations without hesitating
- [ ] Say what you'd do next, and why data comes before a fancier model
- [ ] Quote one specific weight the model learned — `great +10.49`, `not −9.33`, or `not good −3.45`

**The single most important habit:** when you don't know something, say *"I don't know, but here's how I'd find out."* That answer scores better than a confident guess every time.
