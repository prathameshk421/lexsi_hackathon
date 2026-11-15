# Design notes

These notes explain the visual choices in the demo app. I studied webflow.com, linear.app, stripe.com and vercel.com in a browser and read their computed styles. I took ideas only. No logo, text, image, font file or CSS came from them.

## What I saw

| Site | Observation |
| --- | --- |
| Linear | Inter at weights 510 and 590, not 700. Headings 56 px with about -2% letter spacing. Body 15 px on a 24 px line. Grey body text (`#8a8f98`) under near-white headings. Hairline borders of 0.5 px. The product UI itself is the hero image. |
| Stripe | Light weight (300) headings in navy, not black. A two-tone headline: the lead clause is dark and the rest is muted grey. Buttons with a 4 px radius. One accent (indigo) for actions and links. |
| Vercel | Two text greys only (`#ededed` and `#a1a1a1`). A 6 px radius on every control. Borders instead of shadows. Mono type for small meta text. |
| Webflow | One saturated blue for the primary action. Headings at weight 600 with tight letter spacing. A one-line value statement above the fold. |

## Principles for this app

1. **One accent, one neutral scale.** A warm off-white or near-black page, white or graphite surfaces, two text greys and one cobalt accent. The accent marks focus, the selected state and the story rule. Primary buttons are ink, not blue, so colour stays rare.
2. **Hierarchy from weight and grey, not size alone.** The scale is 12, 13, 14, 15, 17, 22 and 40 px. Weights are 400, 500 and 600. Large headings get negative letter spacing. Secondary text is grey, not smaller and smaller.
3. **Hairlines instead of shadows.** Cards, toolbars and tables use 1 px borders in the neutral scale. There are no drop shadows on content. Radii are 4 px for tags, 6 px for controls and 10 px for cards.
4. **The product is the hero.** The headline says the idea in one line ("Same question. Same right answer. Very different cost."). The live comparison sits right under it. A sentence built from the two traces gives the cost gap in numbers before any chart.
5. **The data is the design.** SQL and tool output use a mono face with colour on keywords, strings and numbers. Long output shows six lines, and a native disclosure shows the rest. Meters carry the real value, the optimal value and the ratio in text. Tables use tabular figures and right-aligned numbers.
6. **Meaning never rides on colour alone.** "Correct" and "Wrong" are words. Bars above the optimal cost are orange and hatched, and their ratio is printed. Scripted reference policies have a dashed card border and a "scripted reference" tag, and their chart bars are hatched. A run with a wrong answer gets an outline cost bar and the words "no correct answer", so a cheap failure never looks like a good result.
7. **An 8 px rhythm.** Gaps and padding are 4, 8, 12, 16, 20, 24, 32, 40 and 64 px. Line length stays under about 70 characters for prose.
8. **Short motion.** A new step fades and slides 4 px in 180 ms while a live run streams. Nothing else moves. `prefers-reduced-motion` turns all of it off.

## Practical choices

- **Fonts.** IBM Plex Sans and IBM Plex Mono from one Google Fonts stylesheet. The stacks fall back to system UI and system mono fonts, so the layout holds when the font does not load.
- **Gradio.** A custom `gr.themes.Base` theme sets every colour, border and radius. The CSS restyles the tab bar, the radio groups and the toolbars, and hides the Gradio footer. Results, tables, schema and the playground rows are escaped HTML, because a styled HTML table reads better than the stock data frame and has no hidden-tab paint bug.
- **Themes.** Colour tokens live on `:root` and are redefined under `.dark`, the class Gradio sets from the system preference. All text meets WCAG AA in both themes (checked in the browser on every tab).
- **Layout.** The page is at most 1200 px wide, with 32 px gutters on desktop and 16 px on phones. Two-column areas stack below about 900 px. Touch targets are at least 40 px high.
