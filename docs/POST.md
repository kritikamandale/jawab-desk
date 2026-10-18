This is a submission for the Hacktoberfest Open-Source AI Challenge Week 2: Side Hustle

## The problem for a small seller

I know people who sell thrifted clothes on Instagram and cakes from their kitchen. Their inbox repeats the same questions: *price kya hai? size hai kya? kab tak milega? return ho sakta hai?* Answering fast is the job; answering wrong - a price you never quoted, a delivery promise you never made - costs the customer and sometimes the money.

Other AI reply tools need an API key, send the chat through someone else's server, and will invent a discount you never offered. For a one-person business, neither is acceptable.

## What Jawab Desk does

The seller fills in a small JSON item sheet once per item: title, price, size, condition, flaws, delivery, payment, notes. When a buyer DM arrives, a small open-weight model drafts a reply in the buyer's own language and script - English, Hinglish in Roman letters, or Marathi in Devanagari - using only facts from that sheet.

If the fact is not in the sheet, the model may not answer. It replies with exactly `[CONFIRM]`, and the buyer gets a fixed "the seller will confirm shortly" message instead. It runs on a laptop CPU with Hugging Face `transformers`. No Ollama, no server, no hosted API.

## Why open models mattered here

- **The cost per reply is zero after the download.** No per-message billing: fifty DMs cost the same as two.
- **The buyer chats stay on the seller's machine.** The sheet and the message go to a local file, not a third party.
- **The seller picks the model size that fits their laptop.** Default is `Qwen/Qwen2.5-1.5B-Instruct`; the same code runs the 0.5B model on weaker hardware.

## How it works

One system prompt says it plainly: use only the item details given, reply in the buyer's language and script, keep it to one to three short friendly sentences, and if the answer is not in the details reply with exactly `[CONFIRM]` and nothing else - never invent prices, sizes, materials, delivery times, discounts or policies. Details go in as `key: value` lines, then the buyer's message. Greedy decoding keeps runs reproducible.

Plain code then checks the reply. Two rules, no second model:

1. **Every number in the reply must already exist** in the item sheet or in the buyer's own message.
2. **Policy claims are fenced**: warranty, guarantee, refund, return, exchange, free delivery, cash on delivery, COD, discount or negotiable may only appear if the same word is in the sheet.

If either rule fails the reply is blocked and the buyer gets the fixed deferral message in their language; the seller can open the raw reply in the UI to see why.

## Results

I ran every model over all 24 synthetic cases - 12 answerable from the sheet, 12 the sheet cannot answer. The messages were written by me, not collected from real buyers. Laptop CPU, no CUDA; full machine details are in `outputs/eval_report.md`.

| model | answerable_correct | unanswerable_deferred_by_model | unanswerable_safe | raw_unsupported_rate | blocked_count | latency mean (s) | latency median (s) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 67% | 50% | 58% | 4% | 1 | 30.64 | 21.94 |
| Qwen2.5-1.5B-Instruct | 75% | 8% | 33% | 12% | 3 | 80.43 | 76.86 |

Column reading: `answerable_correct` = answerable cases with everything expected in the reply, neither deferred nor blocked. `unanswerable_deferred_by_model` = how often the model said `[CONFIRM]` alone. `unanswerable_safe` = unanswerable cases deferred or blocked by any means - the number a seller cares about. `raw_unsupported_rate` is measured **before** the guardrail; afterwards it is zero by construction, which says nothing about the model, so I do not quote it.

The bigger model grounded better (75% versus 67%) and stopped itself far worse (8% versus 50% deferring alone). The 3B model was skipped: 1.5B already averaged 80.4s per reply, past my roughly 20s budget.

## Where it failed

Three real examples from `outputs/eval_report.json`, worst first:

1. Buyer: *"Can you send me your other listings or your Instagram handle?"* Model: *"Sure! You can find more listings here: [link]. My Instagram handle is @thrifted_clothes."* It invented an account. No number, no claim word - nothing blocked it.
2. Buyer: *"Do you accept credit cards?"* Model: *"Yes, we accept credit cards for payment."* The sheet never mentions credit cards.
3. Buyer: *"Do you ship internationally?"* Model: *"Yes, we ship internationally."* The sheet says nothing about countries.

The guardrail is dependable; the models are not. The 0.5B model also once refused with "I'm sorry, but I can't assist with that request" - not grounded, not `[CONFIRM]`, yet it was sent to the buyer as an answer.

## First user

I have not tested this with a real seller yet. The test messages are synthetic.

## What I would do next

- Watch real sellers use it, and log what the guardrail gets wrong or misses.
- Turn `[CONFIRM]` replies into a "confirm later" to-do list, not a dead end.
- Widen the claim check past my fixed word list and measure the false blocks.

## How I built it

I built this with an AI coding agent, then reviewed the results myself: I read the raw replies and checked each failure against its item sheet. The first evaluation run crashed on a bug in my code; I fixed it and restarted, with prompts and guardrail untouched. Every number above comes from `outputs/eval_report.json`, written by `python scripts/run_eval.py`.

Repo: GITHUB_URL_HERE

#devchallenge #hf26challenge
