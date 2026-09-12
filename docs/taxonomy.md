# Intent taxonomy: labelling guideline

This file is the single source of truth for intents. It is used twice: as the
instruction sheet for hand-labelling the golden set, and as the few-shot
section of the classifier prompt. `python -m src.classify` renders a compact
view of it for the prompt (`artifacts/classifier_prompt.txt`) from each
intent's Summary line, its first two examples and the tie-break order, so
keep every Summary line on a single line. Change an intent here and both uses
change.

Every quoted message is copied exactly from `data/sample/cases_clean.jsonl`
with its `case_id`. The text is already normalised, so handles appear as
`@USER` and links as `URL`. The messages come from the Customer Support on
Twitter dataset (Kaggle, CC BY-NC-SA 4.0).

The cases quoted here are part of the classifier prompt. Keep them, and any
case sharing their `thread_id` or `dup_group_id`, out of the golden set, or
the classifier is graded on its own examples.

## How to label

1. If the message is not mainly in English, the label is `other_unclear`.
   Decide this first, before looking at the topic.
2. Otherwise, find every intent whose definition the message fits. Read
   `prior_turns` when the message is a reply.
3. If more than one intent fits, use the tie-break order at the end of this
   file.
4. The definitions decide the label, not the cluster a message came from.
   The clusters were only the starting point.

## The intents

| intent | handling | merged from clusters | messages in those clusters (of 4,000) |
|---|---|---|---|
| billing_subscription | escalate | 9, 12, 15, 17, 20 | 713 (17.8%) |
| account_access | split | 6, 21 | 309 (7.7%) |
| playback_failure | auto | 4, 16 | 397 (9.9%) |
| library_playlists | auto | 7, 14 | 354 (8.9%) |
| content_unavailable | auto | 2, 8, 23 | 536 (13.4%) |
| feature_request | auto | 3, 10, 19, 25 | 587 (14.7%) |
| followup_diagnostic | split | 1, 18, 24 | 478 (12.0%) |
| chatter_thanks | auto | 13, part of 5 | 155 (3.9%) + part of 238 |
| other_unclear | escalate | 11, 22, rest of 5 | 233 (5.8%) + rest of 238 |

Shares come from the 4,000-message clustering sample
(`artifacts/clusters_summary.md`) and are estimates only: clusters are impure,
and the definitions move individual messages between intents (resolved
follow-ups from cluster 1 are `chatter_thanks`, for example). Cluster 5 is
split between `chatter_thanks` and `other_unclear`. The golden set measures the
real mix.

Handling: **auto** means the agent may reply without a human, **escalate**
means a human handles it, **split** depends on the message as stated under
the intent.

## billing_subscription

**Summary.** Charges, refunds, payment methods, a paid plan that is not active, Student or promotional pricing, cancelling, and anything about Premium for Family.

**Definition.** Payments and plans: charges that are unexpected, doubled,
after cancelling or at the wrong price; refunds; payment methods that fail or
cannot be updated; a paid plan that is not active (the account still shows
Free or a trial after paying); Student and promotional pricing and
eligibility; cancelling or changing a subscription; and everything about
Premium for Family, including joining, invites, the address check and members
who cannot get Premium. It does not cover problems where Premium is only
mentioned as context and nothing about the payment or plan is wrong (downloads
vanishing on a Premium account are library_playlists), or questions about what
Premium does before buying (feature_request). If a charge, payment or plan
problem appears anywhere in the message, the label is billing_subscription.

**Handling: escalate, always.** Resolving it needs account and payment data
the agent has no access to.

**Examples**
- [case 1800705] "@USER I have been charged for premium membership but it shows that I don't have premium."
- [case 1927438] "@USER I've been charged the full £9.99 for premium even though I'm on the student plan, can someone please help me with this"
- [case 840527] "@USER I'm trying to join Premium for Family but I get this error: Unable to join this Family plan. Can you help me?"

**Near misses**
- [case 2841] "@USER Hello! @USER my premium account downloaded songs keep getting erased every time i log out!" → library_playlists: Premium is only context; the problem is downloads disappearing, and nothing about payment is wrong.
- [case 118853] "@USER hey! Thinking about premium but want to know if album art will show in my car. Will be from downloaded music, not streamed. This is important as it's a deal breaker!!" → feature_request: a question about what a feature does before buying, with no charge or plan problem.

## account_access

**Summary.** Logging in, signing up, password resets, a lost email or Facebook login, and compromised accounts (hacked, or used by someone else).

**Definition.** Getting into an account: logging in, signing up, password
resets and reset emails that do not arrive, an email address or username the
customer can no longer use, Facebook-linked logins that stopped working,
being logged out and unable to get back in, and compromised accounts, meaning
someone else using the account, an unknown device, or an email or password
changed by someone else. It does not cover login trouble for Premium for
Family members or login problems that come with a charge (both
billing_subscription), or messages that mention the account without naming
an access problem (other_unclear).

**Handling: split.** Auto-handle password resets, sign-up problems and login
confusion. Escalate, always, when the account is compromised (hacked, taken
over, used by someone else): recovering it needs identity checks and account
data the agent does not have.

**Examples**
- [case 499281] "Hey @USER I'm locked out of my account and can't reset my password" (auto)
- [case 2059128] "@USER @USER please help, my account has been hacked and they have changed my email address so I can't log in 😬😱" (escalate: compromised)
- [case 2463178] "@USER I deleted my facebook account which was linked to my Spotify account. I can't log anymore into Spotify :( Any help/tips ? Thanks" (auto)

**Near misses**
- [case 12093] "@USER I got kicked off of my own account & cant log back in & y'all are still charging me for premium but I can't even use it😪😪" → billing_subscription: a login problem, but it also reports an ongoing charge, and billing outranks everything.
- [case 2919447] "@USER hello i have problems with my familiar premium plan, my family members just can't login." → billing_subscription: family members who cannot log in are a Premium for Family problem.

## playback_failure

**Summary.** Spotify is broken right now: songs won't play, stop or skip, the app crashes or says it is offline, devices or Connect fail, outages.

**Definition.** Spotify is broken right now: songs will not play, stop, skip,
cut out or play the wrong track; the app or web player crashes, freezes or
says it is offline; a device or Spotify Connect will not play; error codes;
and outage reports or questions such as whether Spotify is down. The customer
expects something to work and it does not. It does not cover songs the app
marks as unavailable (content_unavailable), downloads, playlists or saved
music that disappeared (library_playlists), behaviour that works as designed
but annoys, such as ads (feature_request), or someone else playing music on
the account (account_access).

**Handling: auto.** Status information and standard troubleshooting steps do
not need account or payment data.

**Examples**
- [case 1771326] "@USER I'm having issues playing most songs. Only random ones are playing and showing up."
- [case 1698015] "@USER All songs keep cutting in and out. Updated app and still happening. Only happening to audio from spotify. Please look into this."
- [case 2912445] "@USER Is Spotify down? Can't access anything I haven't downloaded. Can't search. Nothing. Lots of others reporting the same... ?"

**Near misses**
- [case 1937495] "@USER still can't listen to @USER 's new album. keeps saying "This song is unavailable" not happy" → content_unavailable: the app itself says the song is unavailable, which is a catalogue gap, not a playback fault.
- [case 2627027] "@USER @USER hi guys, there's a device connected to/using my Spotify which isn't mine. What can I do to make it stop?" → account_access: it reads like a device problem, but someone else is using the account, so it is compromised and escalates.

## library_playlists

**Summary.** The customer's own collection: playlists, saved music or downloads that vanished, changed, or won't download.

**Definition.** The customer's own collection: playlists that vanished,
changed or keep songs the customer removed; saved songs and albums that
disappeared; downloads that delete themselves or will not download; offline
mode; and being stuck at the download or library limit. It does not cover
requests for playlist or library features that do not exist, such as seeing
who follows a playlist or folders, or asking for the limit to be raised
(feature_request, whatever the topic); playlists lost because someone else
got into the account (account_access); or tracks missing from the catalogue
(content_unavailable).

**Handling: auto.** Recovery steps for playlists and downloads are standard
and do not need account or payment data.

**Examples**
- [case 1080676] "@USER went into spotify today and all my playlists are gone"
- [case 2095206] "@USER my offline music keeps deleting from my device every few weeks. It gets really annoying every time I have to re download again"
- [case 2751287] "@USER I've deleted songs from my playlist but they keep on showing up in my que and they're played. Please fix this issue in the next update"

**Near misses**
- [case 2349666] "@USER It's ridiculous that I cannot see who follows my original playlists. Can you explain why this feature isn't available?" → feature_request: it is about playlists, but nothing is broken or missing; it asks for a feature that does not exist.
- [case 2558849] "Someone else has been using my spotify account and deleted all my playlist @USER URL" → account_access: the playlists vanished because the account is compromised, and account_access outranks library_playlists.

## content_unavailable

**Summary.** Music missing from Spotify or blocked in the customer's country, releases not out yet, Spotify not launched there, requests to add music.

**Definition.** Music the customer wants is not on Spotify, or not for them: a
song, album or artist missing from the catalogue or removed, a release that
has not appeared yet, tracks marked unavailable or greyed out, content blocked
in the customer's country, Spotify itself not launched in their country, and
requests for music to be added. It does not cover songs that are available but
fail to play (playback_failure), music that vanished from the customer's own
playlists or downloads (library_playlists), or how albums and releases are
shown in the app (feature_request).

**Handling: auto.** The answer depends on licensing and regional availability,
which the agent can explain but cannot change, and it needs no account data.

**Examples**
- [case 2284613] "Sadly I can't listen to #Repuation because it is not on @USER 😩"
- [case 1975160] "@USER Hey! do you know why this album isn't accessible on Spotify UK? it's one of my favourites 😭 URL"
- [case 1391113] "@USER you guys need to launch in Kenya. Tired of using VPN!"

**Near misses**
- [case 39761] "@USER what is the presale code for the @USER concert in ATX" → other_unclear: it sits in cluster 2 with the missing-album messages, but it asks for a concert presale code, which fits no intent.
- [case 2606366] "@USER Why can I no longer see the exact date albums and singles came out? I think you guys need to fix that..." → feature_request: the albums are there; what changed is how the app shows release dates.

## feature_request

**Summary.** The product works as designed but the customer wants it changed or asks what it can do: new features and apps, ads, removed features.

**Definition.** The product works as designed and the customer wants it
different, or asks what it can do: requests for new features or apps (Apple
Watch, the iPhone X layout, lyrics); complaints about deliberate behaviour,
such as ads that are too loud, too long or too frequent, recommendations, or
removed features; and questions about what a feature does. This applies
whatever the topic: a request about playlists or the library is
feature_request, not library_playlists. It does not cover anything that is
broken or missing right now (playback_failure, library_playlists,
content_unavailable), ads that appear because a paid plan is not active
(billing_subscription), or requests for music to be added
(content_unavailable).

**Handling: auto.** Nothing needs fixing and no account data is involved: the
agent acknowledges the request and passes the feedback on.

**Examples**
- [case 2361912] "@USER any plans for Apple Watch app? If not, I'll have to switch to Apple Music"
- [case 1917226] "@USER pleaseeeee update your app for iPhone X."
- [case 2429632] "@USER why are your ads so loud??"

**Near misses**
- [case 1304298] "@USER my subscription says I have Spotify Free but I'm being charged $11 every month. I still get ads. So? What's wrong?" → billing_subscription: the ads are a symptom of a paid plan that is not active, and there is a charge.
- [case 1472880] "Your new update doesn't let music play individually and is not even playing music basically at all😡😩 DO ANOTHER UPDATE! @USER" → playback_failure: it demands an update, but what it reports is music not playing.

## followup_diagnostic

**Summary.** A reply whose problem is still open but not named in it: a suggested step didn't help, device or version details, or a pointer to a DM.

**Definition.** A reply whose problem is still open but is not named in the
message itself: the result of a step the brand suggested that did not help,
diagnostic details the brand asked for (device, OS, app version), or a
pointer to a direct message the customer sent. A DM pointer counts even as
the first tweet in a thread, because the conversation is happening in DMs.
The meaning lives in `prior_turns`. It does not cover a reply that names its
problem (label the problem: every topic intent outranks this one), a reply
saying the problem is now fixed (chatter_thanks), or thanks with no open
problem (chatter_thanks).

**Handling: split.** Auto-handle when `prior_turns` give enough context to
continue the conversation. Escalate when the message cannot be interpreted on
its own, including a DM pointer with no prior turns.

**Examples**
- [case 2426245] "@USER Did not work. Still having same issue."
- [case 2218932] "@USER iOS 11.2, Spotify version 8.4.25.906"
- [case 1517702] "@USER Just sent a DM"

**Near misses**
- [case 2095170] "@USER Oooo closing the app and opening it again worked! Thanks!" → chatter_thanks: it is a reply to the brand, but it says the problem is fixed and asks for nothing.
- [case 30600] "@USER @USER I paid for premium and i still havent got premium i have sent a DM @USER" → billing_subscription: it points to a DM, but it names a plan that is not active after paying, and billing outranks everything.

## chatter_thanks

**Summary.** Nothing to resolve: thanks, praise, jokes, sign-offs, and replies saying the problem is now fixed.

**Definition.** Nothing to resolve: thanks, praise, sign-offs, jokes and
reactions, including replies confirming that a problem is now fixed. The
message asks for nothing and reports no open problem. It does not cover thanks
attached to a problem that is still open (followup_diagnostic, or the
problem's own intent when the message names it), or a new question after the
thanks.

**Handling: auto.** There is nothing to resolve; the reply closes the
conversation politely.

**Examples**
- [case 1862690] "Thank you for the great customer service @USER 👏🏽"
- [case 2487419] "@USER That did it! Thanks! I originally just restarted it, but logging out helped. :)"
- [case 2391324] "@USER Thanks for the info. Anyways I love listening to my fav artists on Spotify."

**Near misses**
- [case 634372] "@USER Thanks but doesn't really answer my question??... is it coming back when you do "improve the service?"" → followup_diagnostic: it opens with thanks, but the question is still open and its meaning is in the prior turns.
- [case 89535] "@USER Thanks for response! The app says that music is playing, but no sound comes out. I have to restart to make it work again." → playback_failure: the thanks is politeness; the message reports sound not playing.

## other_unclear

**Summary.** The message alone doesn't say what is needed: not mainly English, content only in a screenshot or link, no problem named, or no intent fits.

**Definition.** The message alone does not tell the agent what to do. Always
label here: messages not mainly in English, whatever they are about (decided
before any other rule); messages whose content is in a screenshot or link we
do not have; messages that name no problem at all; and requests that fit no
intent, such as concert presale codes. It does not cover messages with a URL
whose text says enough on its own (label what the text says), or short
messages that do name a problem, such as asking whether Spotify is down
(playback_failure).

**Handling: escalate, always.** The agent cannot tell what is needed: the
content is in an image, in a language the models do not handle, or not
stated.

**Examples**
- [case 2919313] "@USER what's going on URL"
- [case 2815938] "@USER min, pembayaran spotify premium yang 3 bulan dan 6 bulan hanya untuk akun baru??"
- [case 1780837] "@USER i got problems w my spotify need help pls"

**Near misses**
- [case 1371869] "@USER @USER URL problem fixed. thank you so much for addressing it 🤘⚔️⚡🎶👽👹😉" → chatter_thanks: it has a URL, but the text alone says the problem is fixed.
- [case 1412659] "@USER is Spotify down?" → playback_failure: short, but it names the problem, a possible outage.

## Tie-break rules

**Summary.** A message not mainly in English is other_unclear. Otherwise, if several intents fit, choose the one that comes first in this order.

When a message fits more than one intent, the label is the one that comes
first in this order:

1. billing_subscription
2. account_access
3. playback_failure
4. library_playlists
5. content_unavailable
6. feature_request
7. followup_diagnostic
8. chatter_thanks
9. other_unclear

Why this order:

- **billing_subscription is first because it gates escalation.** A message
  that mentions a charge must never be auto-handled just because it also
  mentions a bug.
- **account_access is next** because a compromised account escalates, and a
  takeover explains problems further down the list, such as playlists
  deleted by an intruder.
- **The failure intents come before feature_request.** Something broken or
  missing now (playback_failure, then library_playlists, then
  content_unavailable) matters more than a wish for the product to change.
- **followup_diagnostic comes after every topic intent.** A follow-up that
  names its problem is labelled by the problem. followup_diagnostic is only
  for replies whose meaning lives in `prior_turns`.
- **chatter_thanks comes after followup_diagnostic** because thanks attached
  to an open problem does not close it.
- **other_unclear is last.** It is the fallback when nothing else fits.

Two rules apply before the order. A message that is not mainly in English is
`other_unclear` whatever its topic; billing_subscription and other_unclear
both escalate, so a charge is still never auto-handled. And a request to change
something that works as designed is feature_request whatever it is about; the
order only decides between intents that genuinely both apply.

Worked examples from the data:

- Case 12093 (account_access near miss above): a login problem plus an
  ongoing charge is billing_subscription.
- Case 30600 (followup_diagnostic near miss above): a DM pointer plus a paid
  plan that is not active is billing_subscription.
- Case 2558849 (library_playlists near miss above): an intruder plus deleted
  playlists is account_access.
- A follow-up that names a playback fault is playback_failure:
  - [case 829593] "@USER Just did both of those, and it still doesn't play. It worked fine yesterday." → playback_failure: a reply to suggested steps, but it names the fault, and playback_failure outranks followup_diagnostic.

## How this taxonomy was induced

The intents were merged by hand from 25 KMeans clusters over 4,000 customer
messages (one per near-duplicate group, embedded with all-MiniLM-L6-v2). See
`artifacts/clusters_summary.md` for sizes and distinctive terms.

KMeans did not only find topics. Clusters 1, 18 and 24 are conversational
positions rather than topics: replies reporting step results, device and
version strings, and DM pointers. Cluster 22 is a language (Indonesian and
Tagalog). Cluster 11 is a modality: a link or screenshot carries the content.
The taxonomy merges these axes deliberately. The positions become
followup_diagnostic, and the language and the modality become other_unclear,
because in each case the message alone does not tell the agent what to do.

Known gap: concert presale code requests have no intent of their own and fall
to other_unclear. A keyword search of `cases_clean.jsonl` finds 131 messages
mentioning presale.
