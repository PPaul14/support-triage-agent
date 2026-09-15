# Labelling quick-reference

One page to keep open beside `python -m src.label`. `docs/taxonomy.md` is the
authority: if this sheet and the guideline ever disagree, the guideline wins.

**First:** a message that is not mainly in English is **9**, whatever it is
about. Otherwise find every intent that fits, and use the tie-break if more
than one does.

## Intent: if the customer message is about X, press N

| key | intent | the message is about |
|---|---|---|
| 1 | billing_subscription | charged, refund, premium not active after paying, student discount, family plan (including family members who can't log in), cancelling, ads despite paying |
| 2 | account_access | can't log in, forgot password, locked out, lost email or Facebook login, hacked or someone else in the account |
| 3 | playback_failure | songs won't play, stop or skip, buffering, app crashes or says offline, "is Spotify down" |
| 4 | library_playlists | playlists gone or changed, downloads disappeared, library wiped |
| 5 | content_unavailable | song, album or artist not available or greyed out, not in my country, "launch in India", a request to add a song or artist |
| 6 | feature_request | asking for a feature that doesn't exist, or what a feature does; "please add..." a feature; complaints about ads working as designed (too loud, too many) |
| 7 | followup_diagnostic | a reply whose problem is still open but not named in it: "tried that, still broken", a version string, "I sent a DM" (a DM pointer counts even as the first tweet) |
| 8 | chatter_thanks | thanks, praise, random chatter, a reply saying the problem is fixed now |
| 9 | other_unclear | screenshot only, not English, no problem named, genuinely can't tell, or fits no intent (such as a concert presale code) |

Borderline cases, from the guideline:

- A follow-up that names its problem gets the problem's key, not 7: a reply
  saying songs still won't play is 3.
- A song the app marks as unavailable is 5, not 3.
- Playlists deleted by someone who got into the account are 2, not 4.
- A message with a URL whose text says enough on its own: label the text,
  not 9.
- Premium only mentioned in passing, with nothing wrong with the payment or
  plan, is not 1.

## Tie-break

**Billing outranks everything:** a message mentioning both a charge and a bug
is billing_subscription. The full order when two intents fit: 1 billing >
2 account > 3 playback > 4 library > 5 content > 6 feature > 7 followup >
8 chatter > 9 other.

## The other prompts, in the order the CLI asks them

| prompt | answer |
|---|---|
| compromised | n, unless someone else got into the account: the customer says so, an unknown device is on it, or someone else changed its email or password. It can be y alongside any intent. |
| escalate | y for 1 billing_subscription, for 2 account_access when compromised, and for 9 other_unclear. y for 7 followup_diagnostic when the earlier turns don't give enough context to continue, including a DM pointer with no earlier turns. n otherwise. |
| escalate reason | asked only when escalate is y: a few words naming why. |
| difficulty | 1 obvious, 2 paused, 3 genuinely torn. |
| notes | skip unless difficulty is 3. |

## Known correction

- **case_id 664584** was labelled compromised = y by mistake. Correct it to
  compromised = n before the golden set is used. Do not edit
  `data/golden/golden_set.jsonl` while labelling is still appending to it.
