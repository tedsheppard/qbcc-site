# Element-page brief — pc-construction-work
**Title:** The work must be construction work or related goods and services
**Breadcrumb:** Requirements of a payment claim
**Anchor id:** `pc-construction-work`
**Output file:** `bif_guide_build/v3/pages/page_pc-construction-work.html`

---
# Global rules (same for every element-page)

1. **Match the format model exactly.** Read the model below and replicate
   the structure: breadcrumb h4, h2 title, opening paragraph, statute
   excerpt, prose body with topical h3 sub-headings, optional case
   extract block, "What this means on a project" bullet list,
   footnotes ordered list at the bottom.

2. **No headings labelling cases.** Do NOT use sub-headings like
   "Key authority", "Leading authority", "Key case" or "Important
   decision". Discuss the case naturally in prose under a topical
   sub-heading instead.

3. **Section reference style.** Always use `s 80` with a space, not
   `s80`. Apply this to every section reference in your prose, your
   footnotes, and your statute-excerpt summary text. The ONE exception
   is the verbatim statute body itself, where the gazetted heading
   number stays as printed (e.g. `<strong>61 Application of chapter</strong>`).

4. **Plain prose default.** Do NOT use `<div class="analysis">`. The
   only special blocks you may use are:
   - `<details class="statute-excerpt" open>` for verbatim statute
   - `<div class="case-excerpt">` for a single judicial extract (use
     this at most ONCE per page, only where the extract earns its
     place — i.e. it's the leading proposition or contains words the
     courts repeatedly quote)
   - `<ul>` / `<ol>` for lists
   - `<ol class="footnotes">` for the footnotes block at the bottom

5. **Verbatim source.** Statutory text, case names, citations, judicial
   block quotes and adjudication reference numbers must be reproduced
   word-for-word from the source. Curly→straight quote conversion is
   fine; everything else is not. Do NOT silently "correct" what looks
   like a typo or wrong cross-reference in the source — reproduce as
   written.

6. **No hallucination.** If a case, section or proposition is not in
   the source bundled below, do not include it.

7. **Audience and register.** Construction project manager / project
   director. Plain English, active voice, "you" framing where natural.
   Avoid "pursuant to", "notwithstanding", "inter alia", "it is
   submitted", "the authorities establish".

8. **Length.** Aim for 400–800 words of prose plus a tight footnote
   block. Don't pad. Don't truncate either if the source has substance.

9. **Case selection.** Pick at most TWO cases to discuss in-text. The
   rest go in the footnotes with citation, judge in parentheses, and a
   one-line gloss of what the case is for. The two in-text cases should
   be either:
   - the leading appellate authority (HCA, QCA, recent NSWCA), or
   - a case whose facts directly illustrate the point.
   Use AT MOST ONE `<div class="case-excerpt">` block on the page,
   reserved for a passage the courts treat as the canonical formulation.

10. **HTML entities.** Use `&mdash;`, `&ndash;`, `&hellip;`,
    `&ldquo;`/`&rdquo;`, `&amp;` rather than literal characters where
    appropriate.

11. **Output ONE `<section id="SLUG">…</section>` block.** No `<html>`,
    `<head>`, `<body>`, `<style>`, `<script>` wrappers. No markdown
    fences. No preamble or explanation outside the HTML.

---

## Format model — replicate this structure exactly

```html
        <section id="pc-construction-contract">
          <h4 class="breadcrumb-heading">Requirements of a payment claim</h4>
          <h2>A construction contract must exist</h2>

          <p>The BIF Act payment regime only switches on if there is a &ldquo;construction contract&rdquo; between you and the other side. Without one, no payment claim can be made and any adjudication that follows has no foundation.</p>

          <details class="statute-excerpt" open>
            <summary>Section 64 BIF Act &mdash; &ldquo;construction contract&rdquo;</summary>
            <div class="statute-body">
              <p><em>construction contract</em> means a contract, agreement or other arrangement under which 1 party undertakes to carry out construction work for, or to supply related goods and services to, another party.</p>
            </div>
          </details>

          <h3>What counts</h3>

          <p>The definition does three things at once. It captures any contract or agreement; it extends to an &ldquo;other arrangement&rdquo; beyond a contract; and it removes any requirement that the agreement be in writing. An oral agreement is enough, and so is a partly-oral / partly-written exchange &mdash; for example, a series of emails confirming scope and price, or a sequence of agreed meeting minutes.<sup><a href="#fn-pc-cc-1">1</a></sup></p>

          <p>The work being undertaken must be construction work, or the supply of related goods and services in connection with construction work. Those concepts are defined separately at s 65 and s 66 and are dealt with on the next two pages.</p>

          <h3>The &ldquo;other arrangement&rdquo; limb</h3>

          <p>The &ldquo;other arrangement&rdquo; words exist to catch dealings that don&rsquo;t quite fit the legal definition of a contract &mdash; for instance, a head contractor&rsquo;s direction to a subcontractor to perform variation work where the parties are operating informally &mdash; without letting in every situation where someone has done work for someone else. Drawing the line has produced the most case law under this part of the Act.</p>

          <p>The current position is that an &ldquo;other arrangement&rdquo; must give rise to either:</p>

          <ul>
            <li>a legally binding obligation (not necessarily a contract &mdash; an estoppel or other source of legal compulsion can suffice, provided some legal consequence flows from it); or</li>
            <li>a &ldquo;concluded state of affairs&rdquo; involving reciprocity &mdash; that is, both parties have accepted mutual rights and obligations relating to payment or price for the works.</li>
          </ul>

          <p>The most recent appellate guidance is <em>Bettar Holdings Pty Ltd trading as Hunt Collaborative v RWC Brookvale Investment Pty Ltd as trustee for Brookvale Development Trust</em> [2025] NSWCA 242. The pleading in <em>Bettar</em> alleged only a common law contract or, in the alternative, an estoppel preventing denial of a contract. The NSW Court of Appeal held that wasn&rsquo;t enough &mdash; what was missing was a separate allegation that the claimant had &ldquo;undertaken&rdquo; to perform work on some basis other than a common law contract. McHugh JA (Bell CJ and Kirk JA agreeing) put the test for an &ldquo;other arrangement&rdquo; in these terms:</p>

          <div class="case-excerpt">
            <p><span class="para-number">[139]</span> a legally binding obligation&hellip; or a concluded state of affairs&hellip; involving some element of reciprocity or acceptance of mutual rights and obligations relating to payment or price for the works.</p>
          </div>

          <p>His Honour expressly declined to resolve the earlier first-instance conflict between <em>Lendlease Engineering Pty Ltd v Timecon Pty Ltd</em> [2019] NSWSC 685 (Ball J &mdash; required a legally binding obligation) and <em>Crown Green Square Pty Ltd v Transport for NSW</em> [2018] NSWSC 1080 (McDougall J &mdash; adopted a less stringent reciprocity-based test),<sup><a href="#fn-pc-cc-2">2</a></sup> because the pleading in <em>Bettar</em> failed under either test. So which test applies in close cases remains open at appellate level.</p>

          <p>The practical consequence is that a bare quantum-meruit claim &mdash; where work has been done but there is no agreement at all about price or payment terms &mdash; is unlikely to qualify. The &ldquo;arrangement&rdquo; requires reciprocal payment obligations, not merely the fact that work was performed.</p>

          <h3>What this means on a project</h3>

          <ul>
            <li>If you have a signed contract, a construction contract exists. Move on to the next element.</li>
            <li>If your engagement is by exchange of emails, purchase orders or agreed meeting minutes, that&rsquo;s still capable of being a construction contract &mdash; provided there is offer, acceptance and an intention to be bound. Document what was agreed and when; you may need to prove it later.</li>
            <li>For variation or out-of-scope work without a written variation order, capture the instruction and your response in writing as soon as possible. Most &ldquo;other arrangement&rdquo; disputes turn on whether there is evidence of agreement on price or rate.</li>
            <li>A bare quantum-meruit case &mdash; just a claim for the value of work done, with no agreement &mdash; will probably fall outside the Act. You&rsquo;d need to show reciprocal payment obligations, not just the fact of work done.</li>
            <li>If you anticipate a challenge to the existence of a contract, plead the arrangement carefully: the material facts of the agreement, the undertaking to carry out work, and the payment obligation. After <em>Bettar</em> the appellate courts are scrutinising those pleadings closely.</li>
          </ul>

          <ol class="footnotes">
            <li id="fn-pc-cc-1">That a written instrument is not required follows from the words &ldquo;contract, agreement or other arrangement&rdquo; in s 64: see <em>Okaroo Pty Ltd v Vos Construction &amp; Joinery Pty Ltd</em> [2005] NSWSC 45 (Nicholas J); <em>Machkevitch v Andrew Building Constructions Pty Ltd</em> [2012] NSWSC 546 (McDougall J).</li>
            <li id="fn-pc-cc-2"><em>Lendlease Engineering Pty Ltd v Timecon Pty Ltd</em> [2019] NSWSC 685, [29]&ndash;[33] (Ball J); <em>Crown Green Square Pty Ltd v Transport for NSW</em> [2018] NSWSC 1080 (McDougall J). For Queensland, <em>Capricorn Quarries Pty Ltd v Devcon Pty Ltd</em> [2010] QSC 28 (Daubney J) is the leading reading of the contract limb of the BIF Act definition.</li>
          </ol>

        </section>

```

---

## Verbatim BIF Act statute

### `statute/chapter_3/section_065.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 65 — Meaning of construction work
(1) Construction work means any of the following work—
(a) the construction, alteration, repair, restoration,
maintenance, extension, demolition or dismantling of
buildings or structures, whether permanent or not,
forming, or to form, part of land;
(b) the construction, alteration, repair, restoration,
maintenance, extension, demolition or dismantling of
any works forming, or to form, part of land, including
walls, roadworks, powerlines, telecommunication
apparatus, aircraft runways, docks and harbours,
railways, inland waterways, pipelines, reservoirs, water
mains, wells, sewers, industrial plant and installations
for land drainage or coast protection;
(c) the installation in any building, structure or works of
fittings forming, or to form, part of land, including
heating, lighting, air-conditioning, ventilation, power
supply, drainage, sanitation, water supply, fire
protection, security and communications systems;
(d) the external or internal cleaning of buildings, structures
and works, so far as it is carried out in the course of their

construction, alteration, repair, restoration, maintenance
or extension;
(e) any operation that forms an integral part of, or is
preparatory to or is for completing, work of the kind
referred to in paragraph(a), (b) or (c), including—
(i) site clearance, earthmoving, excavation, tunnelling
and boring; and
(ii) the laying of foundations; and
(iii) the erection, maintenance or dismantling of
scaffolding; and
(iv) the prefabrication of components to form part of
any building, structure or works, whether carried
out on-site or off-site; and
(v) site restoration, landscaping and the provision of
roadways and other access works;
(f) the painting or decorating of the internal or external
surfaces of any building, structure or works;
(g) carrying out the testing of soils and road making
materials during the construction and maintenance of
roads;
(h) any other work of a kind prescribed by regulation.
(2) To remove doubt, it is declared that construction work
includes building work within the meaning of the Queensland
Building and Construction Commission Act 1991.
(3) However, construction work does not include any of the
following work—
(a) the drilling for, or extraction of, oil or natural gas;
(b) the extraction, whether by underground or surface
working, of minerals, including tunnelling or boring, or
constructing underground works, for that purpose.
Page 97 Current as at 27 April 2025

```

### `statute/chapter_3/section_066.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 66 — Meaning of related goods and services
(1) Related goods and services, in relation to construction work,
means any of the following—
(a) goods of the following kind—
(i) materials and components to form part of any
building, structure or work arising from
construction work;
(ii) plant or materials (whether supplied by sale, hire
or otherwise) for use in connection with the
carrying out of construction work;
(b) services of the following kind—
(i) the provision of labour to carry out construction
work;
(ii) architectural, design, surveying or quantity
surveying services relating to construction work;
(iii) building, engineering, interior or exterior
decoration or landscape advisory services relating
to construction work;
(iv) soil testing services relating to construction work;
(c) goods and services, relating to construction work, of a
kind prescribed by regulation.
(2) In this chapter, a reference to related goods and services
includes a reference to related goods or services.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_065.txt`
```
# Annotated BIF Act source — Section 65
# Chapter: CHAPTER 3 – Progress payments
# Section title: Meaning of construction work
# DOCX paragraphs: 1197-1257

[2 Com-BIFSOPA Heading 1] SECTION 65 – Meaning of construction work
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 65    Meaning of construction work 
[1.3 BIFSOPA level 1 (CDI)] Construction work means any of the following work—
[1.4 BIFSOPA level 2 (CDI)] the construction, alteration, repair, restoration, maintenance, extension, demolition or dismantling of buildings or structures, whether permanent or not, forming, or to form, part of land; 
[1.4 BIFSOPA level 2 (CDI)] the construction, alteration, repair, restoration, maintenance, extension, demolition or dismantling of any works forming, or to form, part of land, including walls, roadworks, powerlines, telecommunication apparatus, aircraft runways, docks and harbours, railways, inland waterways, pipelines, reservoirs, water mains, wells, sewers, industrial plant and installations for land drainage or coast protection;
[1.4 BIFSOPA level 2 (CDI)] the installation in any building, structure or works of fittings forming, or to form, part of land, including heating, lighting, air-conditioning, ventilation, power supply, drainage, sanitation, water supply, fire protection, security and communications systems; 
[1.4 BIFSOPA level 2 (CDI)] the external or internal cleaning of buildings, structures and works, so far as it is carried out in the course of their construction, alteration, repair, restoration, maintenance or extension;
[1.4 BIFSOPA level 2 (CDI)] any operation that forms an integral part of, or is preparatory to or is for completing, work of the kind referred to in paragraph (a), (b) or (c), including— 
[1.5 BIFSOPA level 3 (CDI)] site clearance, earthmoving, excavation, tunnelling and boring; and
[1.5 BIFSOPA level 3 (CDI)] the laying of foundations; and
[1.5 BIFSOPA level 3 (CDI)] the erection, maintenance or dismantling of scaffolding; and
[1.5 BIFSOPA level 3 (CDI)] the prefabrication of components to form part of any building, structure or works, whether carried out on-site or off-site; and
[1.5 BIFSOPA level 3 (CDI)] site restoration, landscaping and the provision of roadways and other access works;
[1.4 BIFSOPA level 2 (CDI)] the painting or decorating of the internal or external surfaces of any building, structure or works; 
[1.4 BIFSOPA level 2 (CDI)] carrying out the testing of soils and road making materials during the construction and maintenance of roads; 
[1.4 BIFSOPA level 2 (CDI)] any other work of a kind prescribed by regulation.
[1.3 BIFSOPA level 1 (CDI)] To remove doubt, it is declared that construction work includes building work within the meaning of the Queensland Building and Construction Commission Act 1991. 
[1.3 BIFSOPA level 1 (CDI)] However, construction work does not include any of the following work— 
[1.4 BIFSOPA level 2 (CDI)] the drilling for, or extraction of, oil or natural gas; 
[1.4 BIFSOPA level 2 (CDI)] the extraction, whether by underground or surface working, of minerals, including tunnelling or boring, or constructing underground works, for that purpose.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 65.1    ‘Construction work’
[2.4 Com-BIFSOPA CDI Normal Body Text] A construction contract may contain both ‘construction work’ and work that is not ‘construction work’. Refer to the definition of ‘construction contract’ within section 64, above.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Capricorn Quarries Pty Ltd v Inline Communication Construction Pty Ltd [2012] QSC 388; [2013] 2 Qd R 1, Jackson J rejected a submission that BCIPA should be given a ‘remedial’ construction. His Honour held that, in construing the definitions of ‘construction contract’, ‘construction work’ and ‘related goods and services’, a ‘natural’ construction should be given and that:
[2.5 Com-BIFSOPA Normal Body Quote] Having regard to the context as discussed, in my view, a “natural” construction of the relevant definitions of BCIPA in this case is to be preferred to an approach which seeks to extend the operation of BCIPA by a “liberal interpretation” to be engaged in with the purpose of increasing the width of the class of persons who are entitled to the benefit of a payment claim and correspondingly increasing the width of the class of persons who are subject to BCIPA’s restriction and obligations.
[2.5 Com-BIFSOPA Normal Body Quote] The language chosen by Parliament to define “construction contract”, “construction work” and “related goods and services” has no purpose other than to draw the line between who is in and who is out of those classes.  There seems to be little logic in seeking to stretch that language either way.  In saying this, I take a “natural” construction to be that arrived at by the usual process of the application of the common law of statutory interpretation, as affected by statute, but without a presumptive approach.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Walter Construction Group Ltd v CPL (Surry Hills) Pty Ltd [2003] NSWSC 266, Nicholas J rejected a submission that an extension of time claim did not relate to ‘construction work’.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Patrick Stevedores Operations (No 2) Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2014] NSWSC 1413, Ball J held that neither plant and materials ordered by the claimant to comply with its contractual obligations, nor the costs of removing equipment from the site, were ‘construction work under the contract’ or the supply of ‘related goods or services’.
[2.2 Com-BIFSOPA Heading 3] 65.2    Section 65(1) – to form part of land
[2.4 Com-BIFSOPA CDI Normal Body Text] In Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd [2011] QSC 345, Fryberg J held that the phrase ‘works to form part of land’ is ‘evidently a phrase of wide meaning’ (at [28]). In so doing, his Honour rejected a submission that the phrase ‘forming or to form part of land’ suggests an intention that the works must be something added to the land or intended to be added to the land. His Honour also held that the section ‘should be given a practical interpretation which will minimise potential areas of dispute’ (at [32]).
[2.2 Com-BIFSOPA Heading 3] 65.3    Section 65(1) - ‘forming, or to form, part of land’ – mining leases
[2.4 Com-BIFSOPA CDI Normal Body Text] In J & D Rigging Pty Ltd v Agripower Australia Ltd & Ors [2013] QCA 406, the Court of Appeal held that an ‘ordinary’ meaning should be given to the word ‘land’. Applegarth J (Holmes JA and Boddice J agreeing) said:
[2.5 Com-BIFSOPA Normal Body Quote] BCIPA reallocates financial risk between the parties to a construction contract to which it applies. Section 10 should be interpreted so that its ordinary words are able to be applied by parties to determine whether or not a contract is subject to BCIPA. If possible, the statute should be interpreted so that it is capable of being applied in a practical way by parties to a construction contract or a proposed construction contract. The ordinary meaning of the words in s 10 anticipates a practical inquiry into the physical relationship between an item and land, by asking in the case of s 10(1)(a) whether the building or structure forms, or is to form, part of land. In the case of the demolition or dismantling of a building or structure, the inquiry is whether the building or structure forms part of land. The inquiry is into the physical state of things, not the intention of parties at the time the building or structure was constructed, possibly many years earlier.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] The word “land” in s 10 has the extended meaning contained in the Acts Interpretation Act as well as its ordinary meaning. The expression “forming, or to form part of land” uses the word “form” in its ordinary sense.
[2.5 Com-BIFSOPA Normal Body Quote] The words of s 10 do not call for an inquiry into whether the plant formed part of land according to common law rules about the ownership of fixtures. They do not import a requirement that the plant be owned by the owner of the land upon which it was constructed as a result of this having been the objective intention of certain parties. Requirements of the law of real property about ownership of things affixed to land are not imported into s 10. Instead, the degree of annexation will be relevant to the issue of whether or not a thing forms part of land.
[2.2 Com-BIFSOPA Heading 3] 65.4    Section 65(1)(b)
[2.4 Com-BIFSOPA CDI Normal Body Text] In Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd [2011] QSC 345, Fryberg J rejected a submission that works under this paragraph should be intended to be permanent (at [31]).
[2.4 Com-BIFSOPA CDI Normal Body Text] In BCS Infrastructure Support Pty Ltd v Jones Lang Lasalle (NSW) Pty Ltd [2020] VSC 739, Justice Stynes found that the maintenance of: baggage conveyor systems, aerobridges, check-in kiosks and baggage return carousels, fell within the meaning of construction work under section 5(1)(a) of the NSW SOP Act (the equivalent of section 65(1)(b) of the BIF Act) on the basis that the significant structures were permanently attached to the terminal building. As such, Stynes J determined that the construction contract in issue did constitute a construction contract for the purposes of section 4 of the SOP Act (the equivalent of section 64 of the BIF Act).
[2.2 Com-BIFSOPA Heading 3] 65.5    Section 65(1)(e)
[2.3 Com-BIFSOPA Heading 4] ‘[A]ny operation that forms an integral part of, or is preparatory to or is for completing, work of the kind…’
[2.4 Com-BIFSOPA CDI Normal Body Text] Paragraph (e) of subsection 10(1) of the Act is defined by reference to paragraphs (a), (b) and (c) of this subsection 10(1). As to this relationship, in Wiggins Island Coal Export Terminal Pty Ltd v Monadelphous Engineering Pty Ltd & Ors [2015] QSC 307, Philip McMurdo J said that what is important is:
[2.5 Com-BIFSOPA Normal Body Quote] the close connection between an operation within paragraph (e) and construction work of a kind within paragraphs (a), (b) or (c).  An operation falls within paragraph (e) only because of the part it plays in construction work which, according to the terms of paragraphs (a), (b) and (c), involves a connection with certain land. 
[2.3 Com-BIFSOPA Heading 4] Section 65(1)(e)(i)
[2.4 Com-BIFSOPA CDI Normal Body Text] In Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd [2011] QSC 345, Fryberg J held that clearing and grubbing work to fall under this paragraph. See also Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd [2012] QCA 276; [2013] 2 Qd R 75.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Cadia Holdings Pty Ltd v Downer EDI Mining Pty Ltd [2020] NSWSC 1588, Stevenson J held that preparatory work that is not for the actual purpose of extracting minerals, falls within the meaning of construction work and is not excluded by the “mining exception” set out in section 5(2) of the NSW SOP Act (the equivalent to section 65(1)(e) of BIFSOPA). In that case, Cadia Holdings Pty Ltd (Cadia) engaged Downer EDI Mining Pty Ltd (Downer) to carry out “tunneling or boring” as well as “constructing underground works”, so as to construct an access to the proposed undercut and extraction levels of the mine. In challenging an adjudication determination made in favour of Downer, Cadia contended that the scope of works under the contract did not fall within the definition of “construction work” and was expressly excluded under the “Mining Exception” in s 5(2)(b) of the SOP Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] In determining that the scope of works under the contract was “construction work”, the NSW Supreme Court held that the Mining Exception was to be construed narrowly to benefit the claimant (i.e. Downer); that is, the legislative intention that tunneling, boring or construction of underground works referred to in the Mining Exception must be for the actual purpose of extracting minerals. As such, where the object and purpose of a contract is not to cause the contractor to undertake work “for the purpose of actually extracting minerals”, the Mining Exception will not apply.
[2.3 Com-BIFSOPA Heading 4] Section 65(1)(e)(iv)
[2.4 Com-BIFSOPA CDI Normal Body Text] In Wiggins Island Coal Export Terminal Pty Ltd v Monadelphous Engineering Pty Ltd & Ors [2015] QSC 307, Philip McMurdo J held that a prefabricated shiploader and ‘tripper’ to be installed to a wharf on a coal export terminal were properly characterised as components to form part of that wharf and within the meaning of section 10(1)(e) of the Act. 
[2.2 Com-BIFSOPA Heading 3] 65.6    Section 65(3)
[2.3 Com-BIFSOPA Heading 4] Coal
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been held that, in the context of section 10(3) of BCIPA, coal is a ‘mineral’ for the purposes of section 65(3) of the Act,: Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd [2011] QSC 345, [39] (Fryberg J).
[2.3 Com-BIFSOPA Heading 4] ‘Extraction’
[2.4 Com-BIFSOPA CDI Normal Body Text] In Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd [2011] QSC 345, Fryberg J adopted a narrow construction to section 10(3) of BCIPA. His Honour held (at [42]): 
[2.5 Com-BIFSOPA Normal Body Quote] I accept that the work performed by Warren was a necessary part of opening the coal mines. But that is not the issue. The exemption given by s 10(3)(b) is not expressed to apply to work done for the purpose of opening or as preparatory to operating a mine. The words used are much more limited than that. They focus purely on the process of extraction. In my judgment the ordinary meaning of the word, considered in isolation, does not apply to the work done by Warren.
[2.4 Com-BIFSOPA CDI Normal Body Text] This approach was adopted by Douglas J in HM Mire Pty Ltd v National Plant and Equipment Pty Ltd [2012] QSC 4 at [13]. Referring to Theiss, Douglas J said:
[2.5 Com-BIFSOPA Normal Body Quote] I agree with his Honour’s reasons dealing with the proper construction of the section. The facts of this case are also similar to the facts then being considered by his Honour. It also seems to me to be necessary to construe s 10(3) in the context set by s 10(1) which describes a relatively broad set of circumstances amounting to construction work. It would detract unnecessarily from the apparent purpose of the legislation and the normal understanding of s 10(1) and its hierarchy in the section to extend the meaning of “extraction of minerals” to cover work associated with such extraction where the legislature, as Mr Ambrose submitted, could readily have made such a purpose clear by the use of familiar language of wider meaning than this phrase. There is no reason why s 10(3) should be read so as to displace or render nugatory the meaning of s 10(1).
[2.4 Com-BIFSOPA CDI Normal Body Text] This approach was approved by the Court of Appeal in Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd [2012] QCA 276; [2013] 2 Qd R 75, 91 [62]-[63] (Philippides J, Holmes and White JJA agreeing).

```

### `annotated/section_066.txt`
```
# Annotated BIF Act source — Section 66
# Chapter: CHAPTER 3 – Progress payments
# Section title: Meaning of related goods and services
# DOCX paragraphs: 1258-1292

[2 Com-BIFSOPA Heading 1] SECTION 66 – Meaning of related goods and services
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 66    Meaning of related goods and services 
[1.3 BIFSOPA level 1 (CDI)] Related goods and services, in relation to construction work, means any of the following— 
[1.4 BIFSOPA level 2 (CDI)] goods of the following kind— 
[1.5 BIFSOPA level 3 (CDI)] materials and components to form part of any building, structure or work arising from construction work;
[1.5 BIFSOPA level 3 (CDI)] plant or materials (whether supplied by sale, hire or otherwise) for use in connection with the carrying out of construction work; 
[1.4 BIFSOPA level 2 (CDI)] services of the following kind—
[1.5 BIFSOPA level 3 (CDI)] the provision of labour to carry out construction work; 
[1.5 BIFSOPA level 3 (CDI)] architectural, design, surveying or quantity surveying services relating to construction work;
[1.5 BIFSOPA level 3 (CDI)] building, engineering, interior or exterior decoration or landscape advisory services relating to construction work;
[1.5 BIFSOPA level 3 (CDI)] soil testing services relating to construction work; 
[1.4 BIFSOPA level 2 (CDI)] goods and services, relating to construction work, of a kind prescribed by regulation.
[1.3 BIFSOPA level 1 (CDI)] In this chapter, a reference to related goods and services includes a reference to related goods or services.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 66.1    A ‘natural’ construction
[2.4 Com-BIFSOPA CDI Normal Body Text] In Capricorn Quarries Pty Ltd v Inline Communication Construction Pty Ltd [2012] QSC 388; [2013] 2 Qd R 1, Jackson J rejected a submission that the Act should be given a ‘remedial’ construction. His Honour held, that, in construing the definitions of ‘construction contract’, ‘construction work’ and ‘related goods and services’, a ‘natural’ construction should be given. His Honour held:
[2.5 Com-BIFSOPA Normal Body Quote] Having regard to the context as discussed, in my view, a “natural” construction of the relevant definitions of BCIPA in this case is to be preferred to an approach which seeks to extend the operation of BCIPA by a “liberal interpretation” to be engaged in with the purpose of increasing the width of the class of persons who are entitled to the benefit of a payment claim and correspondingly increasing the width of the class of persons who are subject to BCIPA’s restriction and obligations.
[2.5 Com-BIFSOPA Normal Body Quote] The language chosen by Parliament to define “construction contract”, “construction work” and “related goods and services” has no purpose other than to draw the line between who is in and who is out of those classes.  There seems to be little logic in seeking to stretch that language either way.  In saying this, I take a “natural” construction to be that arrived at by the usual process of the application of the common law of statutory interpretation, as affected by statute, but without a presumptive approach.
[2.2 Com-BIFSOPA Heading 3] 66.2    ‘Supply’
[2.4 Com-BIFSOPA CDI Normal Body Text] In Patrick Stevedores Operations (No 2) Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2014] NSWSC 1413, Ball J held that neither plant and materials ordered by the claimant to comply with its contractual obligations, nor the costs of removing equipment from the site, were ‘construction work under the contract’ or the supply of ‘related goods or services’. His Honour held that plant and materials ordered were to put the claimant ‘in a position where it could comply with its contractual obligations’, but it was not by itself, the supply of related goods or services.
[2.2 Com-BIFSOPA Heading 3] 66.3    Section 66(1)(a) – ‘for use in connection with’
[2.4 Com-BIFSOPA CDI Normal Body Text] This phrase was considered in Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd [2011] QSC 345. Fryberg J held:
[2.5 Com-BIFSOPA Normal Body Quote] In my judgment, “for use in connection with” is not satisfied simply by proving that the plant or materials supplied were used in connection with the carrying out of construction work.  Warren's submission to that effect should be rejected. “For” is a word of wide denotation.  In the present context it carries a purposive meaning.  It suggests that the phrase must be satisfied at the outset of the transaction, before the plant or materials are used.  Evidence of the use to which plant or materials were put may support an inference as to the purpose for which they were at that time to be used, but such evidence could not satisfy the phrase without such an inference.
[2.5 Com-BIFSOPA Normal Body Quote] That conclusion is supported by the manner in which “related goods and services” is used in the Act. The most important use is in the definition of construction contract, a definition which relevantly refers to a “contract, agreement or other arrangement”. As noted above, in most other contexts in which the expression is used it is used to refer to related goods and services supplied under a construction contract. That suggests that the contract or arrangement will supply an important part of the evidence by which the proposed use of the plant or materials is to be identified. The use must be able to be identified at the time of the contract or arrangement or possibly, if the supply takes place at an earlier time, at the time of the supply.
[2.4 Com-BIFSOPA CDI Normal Body Text] This was adopted by Jackson J in Capricorn Quarries Pty Ltd v Inline Communication Construction Pty Ltd [2012] QSC 388; [2013] 2 Qd R 1, 11 [49].
[2.2 Com-BIFSOPA Heading 3] 66.4    ‘Services’
[2.4 Com-BIFSOPA CDI Normal Body Text] In Sheppard Homes Pty Ltd v FADL Industrial Pty Ltd [2010] QSC 228, Fryberg J considered that the supply of a licence for drawings does not fall within the definition of services.
[2.4 Com-BIFSOPA CDI Normal Body Text] The agreement in question in Sheppard Homes Pty Ltd v FADL Industrial Pty Ltd [2010] QSC 228 was an agreement between a consultant and a builder under which the consultant provided a licence to the builder to use certain plans for building work. Fryberg J held that the agreement was not the provision of related services under the Act.
[2.3 Com-BIFSOPA Heading 4] Architectural services 
[2.4 Com-BIFSOPA CDI Normal Body Text] Architectural services have been held to fall under the definition of ‘related goods and services’: Peter’s of Kensington v Seeksucker Pty Ltd [2008] NSWSC 897.
[2.3 Com-BIFSOPA Heading 4] Project management services
[2.4 Com-BIFSOPA CDI Normal Body Text] Project management services have also been held to fall under the definition of ‘related goods and services’: Biseja Pty Ltd v NSI Group Pty Ltd [2006] NSWSC 835.

```
