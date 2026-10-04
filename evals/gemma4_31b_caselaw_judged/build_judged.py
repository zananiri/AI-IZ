"""Write the Claude judgments for the gemma4_31b_caselaw run (Prompt A rubric) as score.py input."""
import json
from pathlib import Path

O = Path(__file__).parent
G = O.parent / "gemma4_31b_caselaw_results/AI-IZ/data/legal/eval_caselaw/gemma4_31b_caselaw/gemma4_31b"

# id: (correctness, grounding, note[, "h" = hallucination])
A = {
    'IL-483': (2, 'grounded', 'Refuses, names the offence, points to the lawful insolvency route.'),
    'IL-484': (2, 'n/a', 'Says future legislation cannot be known.'),
    'IL-486': (2, 'grounded', 'No 30 days; reasonable time, s.20.'),
    'IL-487': (2, 'grounded', 'One month, s.3(3).'),
    'IL-493': (2, 'grounded', 'Corrects the false premise.'),
    'IL-496': (2, 'n/a', 'No access to personal case decisions.'),
    'IL-108': (1, 'grounded', "s.14 right, but claims there is no law called 'חוק התרופות'."),
    'IL-110': (2, 'grounded', 's.5 with (b) and (c).'),
    'IL-240': (2, 'grounded', 's.5 and the periods.'),
    'IL-388': (2, 'grounded', 's.8.'),
    'IL-389': (2, 'grounded', 's.20(b).'),
    'IL-391': (2, 'grounded', 's.2A2 temporary provision; does not name s.12 of the amending law.'),
    'IL-432': (2, 'grounded', 's.8(a) and the four conditions.'),
    'IL-036': (2, 'grounded', 'All four contract types.'),
    'IL-040': (2, 'grounded', 'Both elements of s.18.'),
    'IL-230': (2, 'grounded', 's.1 definition plus transport-risk case law.'),
    'IL-274': (2, 'grounded', 's.11(a) plus case law on warning the employer first.'),
    'IL-276': (2, 'grounded', 'Only the labour court may hear these matters; s.24.'),
    'IL-325': (2, 'grounded', 'Both forms of recklessness, s.20(a)(2).'),
    'IL-327': (0, 'misgrounded', 'Says no statutory reasonableness requirement exists; misses s.34P (34טז).'),
    'IL-420': (2, 'grounded', 'ss.194, 198, court approval.'),
    'IL-452': (1, 'grounded', 'Disclosure s.6, s.7, s.23; no 30-day payment, interest or limitation (2/5).'),
    'IL-453': (2, 'grounded', 'Self-defence, immediacy, reasonableness (34J1(b)), offences; no property-vs-body point (4/5).'),
    'IL-455': (1, 'grounded', 'Fiduciary duty s.8 and rescission s.9; no authority limits, s.10 reliance or fraud.'),
    'IL-456': (0, 'misgrounded', 'Gift exclusion right, but says the completed gift can be revoked under s.5(c); no apartment sharing or other remedies.'),
    'IL-461': (1, 'grounded', 'Unfair terms, cancellation fee, jurisdiction clause; no consumer cancellation, misleading renewal or class action.'),
    'IL-468': (1, 'grounded', 'Retaliation and employer duties; no defamation, good faith or labour court.'),
    'IL-470': (1, 'grounded', 'Privacy exception s.9 and public interest s.10; no partial disclosure or petition s.17.'),
    'IL-032': (1, 'grounded', 's.26 and party practice first; no burden of proof or certainty (s.2).'),
    'IL-087': (1, 'grounded', 'Breach and s.10 price difference; distress left uncertain (no s.13), no deposit, no mitigation.'),
    'IL-135': (2, 'grounded', 'ss.25H, 25N: no contracting out; clause invalid.'),
    'IL-136': (1, 'grounded', 's.10 applied but flatly for the buyer; ignores the forgery case-law question and buyer diligence.'),
    'IL-223': (0, 'misgrounded', 'Says the Defamation Law is not in the material, dates it 1988, leans on Penal Law s.168; no ss.1, 2, 14, 15, 7A.', 'h'),
    'IL-267': (1, 'misgrounded', 'Notice pay and August wage only; misses severance (~36,000), s.14, late wage, hearing, vacation; Wage Protection Law dated 1971, s.25.', 'h'),
    'IL-268': (1, 'grounded', 's.9(a), 6 months, forced resignation; no Equal Opportunities claim, no documenting advice.'),
    'IL-269': (1, 'grounded', 'Overtime rates right but built on the 10-hour restaurant exception; no weekly rest or rest-day pay.'),
    'IL-317': (1, 'grounded', 'Murder s.300, negligence s.304, provocation s.301B; misses reckless killing s.301C and aggravating circumstances.'),
    'IL-321': (1, 'grounded', 'Release after 9 months s.61 and Supreme Court extension; 90 days misplaced; no 7 Oct law.'),
    'IL-371': (1, 'grounded', 'Default judgment reg.130; no 60-day rule, set-aside or MAHUT.'),
    'IL-415': (1, 'grounded', 'Privacy ss.2(4), 2(11), 4; no defences, defamation or compensation without proof.'),
    'IL-016': (2, 'grounded', 'Misleading by non-disclosure, s.15.'),
    'IL-024': (2, 'grounded', 'Restitution s.21.'),
    'IL-075': (2, 'grounded', 'Personal service exception s.3(2).'),
    'IL-126': (2, 'grounded', 'Writing requirement s.8.'),
    'IL-164': (2, 'grounded', 'ss.5, 6; parents may cancel.'),
    'IL-168': (2, 'grounded', 'Disqualified, s.5.'),
    'IL-172': (2, 'grounded', 'ss.18, 19.'),
    'IL-212': (2, 'grounded', 'Absolute liability s.2(c).'),
    'IL-213': (2, 'grounded', 's.7(1).'),
    'IL-214': (2, 'grounded', 's.13.'),
    'IL-217': (2, 'grounded', 's.7A.'),
    'IL-219': (2, 'grounded', 's.9.'),
    'IL-262': (2, 'grounded', '125% / 150%, s.16.'),
    'IL-306': (0, 'misgrounded', 'Says yes, prosecution after consulting a probation officer (Youth Law); misses the age-12 rule s.34F.'),
    'IL-309': (2, 'grounded', 's.34J.'),
    'IL-368': (2, 'grounded', 'Evacuee ballot options.'),
    'IL-409': (2, 'grounded', 'Court approval s.198.'),
    'IL-410': (2, 'grounded', 'Profit and solvency tests s.302.'),
    'IL-001': (2, 'grounded', 'Offer and acceptance s.1.'),
    'IL-011': (2, 'grounded', 's.26.'),
    'IL-014': (2, 'grounded', 's.41.'),
    'IL-061': (1, 'grounded', "All three remedies, but claims no law is called 'חוק התרופות'."),
    'IL-062': (2, 'grounded', 'All four exceptions s.3.'),
    'IL-113': (2, 'grounded', 's.7.'),
    'IL-115': (2, 'grounded', 's.9.'),
    'IL-153': (2, 'grounded', 'Parents, s.14.'),
    'IL-202': (2, 'grounded', 's.41.'),
    'IL-205': (2, 'grounded', 's.13.'),
    'IL-245': (2, 'grounded', 's.1(a).'),
    'IL-247': (2, 'grounded', 'One month.'),
    'IL-248': (2, 'grounded', 'One day per month.'),
    'IL-253': (2, 'grounded', '9th day.'),
    'IL-254': (2, 'grounded', 'All grounds s.2(a).'),
    'IL-256': (2, 'grounded', 'Regional labour court s.24.'),
    'IL-293': (2, 'grounded', 's.24.'),
    'IL-294': (2, 'grounded', 'Age 12, s.34F.'),
    'IL-296': (2, 'grounded', 'Mandatory life, s.301A.'),
    'IL-297': (2, 'grounded', '12 years.'),
    'IL-300': (1, 'grounded', '24 hours stated, but concludes "usually 24 to 48 hours".'),
    'IL-353': (2, 'grounded', '18 and 21.'),
    'IL-060': (2, 'grounded', 'Treated as not having agreed.'),
    'IL-198': (0, 'ungrounded', 'Says the material lacks the date; no Amendment 18 (2016/2017).'),
    'IL-347': (0, 'grounded', 'Calls the premise false; no check-the-current-version advice.'),
    'IL-392': (2, 'grounded', 'From 1.1.2021; 1984 regulations before.'),
}
assert len(A) == 84


def write(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for i, v in items.items():
            f.write(json.dumps({"id": i, "correctness": v[0], "grounding": v[1], "hallucination": len(v) > 3,
                                "key_points_hit": [], "needs_review": False, "note": v[2]}, ensure_ascii=False) + "\n")


write(O / "judged_caselaw.jsonl", A)
# no_caselaw answers are mostly identical to caselaw ones; differences re-read by hand
bids = [json.loads(l)["id"] for l in open(G / "no_caselaw/bulk500/answers_rag.jsonl", encoding="utf-8")]
B = {i: A[i] for i in bids if i in A}
B["IL-274"] = (1, "grounded", "s.11(a) restated; no meaning of the worsening, no warn-the-employer rule.")
write(O / "judged_no_caselaw.jsonl", B)
print(len(A), len(B))
