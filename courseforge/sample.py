"""A small fictional sample textbook, generated in memory, for trying CourseForge AI.

Every fact in it is invented ("glowmoss" and "dimwings" on the imaginary planet Orvane).
It is not taken from any real textbook, so it is safe to use in demonstrations.
"""

from __future__ import annotations

import pymupdf

SAMPLE_FILENAME = "fictional-sample-textbook-orvane.pdf"
SAMPLE_TITLE = "The Living Cliffs of Orvane (fictional sample textbook)"

TITLE_PAGE = (
    "The Living Cliffs of Orvane\n"
    "A fictional sample textbook created for the CourseForge AI demonstration.\n"
    "Every place, plant and animal in this book is invented."
)

CHAPTER_1 = [
    "Chapter 1: The Glowmoss of Orvane\n"
    "Glowmoss is a fictional plant that grows on the cliffs of the imaginary planet Orvane. "
    "The moss produces a soft blue light at night by storing sunlight during the day. "
    "Each moss colony forms a thick carpet of tiny leaves called lumenfronds. "
    "The lumenfronds absorb sunlight and convert it into stored glow energy. "
    "For example, a colony that receives six hours of sunlight can glow for the whole night.",

    "Glowmoss spreads by releasing silver spores into the cliff winds. "
    "The spores drift for many days before they settle on wet stone. "
    "A spore only grows when the stone stays damp for a full week. "
    "Young colonies glow weakly because their lumenfronds are still small. "
    "Farmers on Orvane collect the spores to start new colonies near their homes.",

    "The glow of the moss attracts night insects called dimwings. "
    "The dimwings carry spores from one cliff to another while they feed. "
    "In return, the moss provides the dimwings with a sweet nectar. "
    "Scientists in the story describe this partnership as a form of mutual support. "
    "Without the dimwings, the moss would spread much more slowly across the cliffs.",

    "Cold winters reduce the amount of stored glow energy in each colony. "
    "During long winters, the moss enters a resting state and stops releasing spores. "
    "The colony wakes again when the cliffs warm and the stone becomes damp. "
    "A healthy colony can survive many winters by resting through the coldest months. "
    "Farmers protect their colonies by covering them with woven grass during storms.",
]

CHAPTER_2 = [
    "Chapter 2: Dimwings and the Night Cliffs\n"
    "Dimwings are fictional night insects that live only on the cliffs of Orvane. "
    "Each dimwing has four pale wings that reflect the blue glow of the moss. "
    "The insects sleep inside cracks in the stone during the day. "
    "At dusk, a dimwing leaves its crack and follows the brightest patches of glowmoss. "
    "Young dimwings stay close to their nest until their wings grow strong.",

    "Dimwings feed on the sweet nectar that glowmoss colonies produce at night. "
    "While a dimwing feeds, silver spores stick to the fine hairs on its legs. "
    "When the dimwing flies to another cliff, some of these spores fall onto damp stone. "
    "In this way, the insects help new glowmoss colonies begin far from the parent colony. "
    "For example, a single dimwing can carry spores across three cliffs in one night.",

    "Strong winds make it hard for dimwings to fly between the cliffs. "
    "During windy nights, the insects stay near their nests and feed on nearby colonies. "
    "Farmers on Orvane plant rows of tall grass to slow the wind near their colonies. "
    "The grass gives the dimwings shelter, so they visit the farm colonies more often. "
    "Healthy dimwing populations therefore help farmers grow larger glowmoss colonies.",
]


def sample_pdf_bytes() -> bytes:
    """Build the sample PDF: a title page, then two chapters listed in its table of contents."""
    pages = [TITLE_PAGE] + CHAPTER_1 + CHAPTER_2
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 770), text, fontsize=12)
    chapter_2_start = 2 + len(CHAPTER_1)
    doc.set_toc([[1, "Chapter 1: The Glowmoss of Orvane", 2],
                 [1, "Chapter 2: Dimwings and the Night Cliffs", chapter_2_start]])
    doc.set_metadata({"title": SAMPLE_TITLE, "author": "CourseForge AI demo (fictional content)"})
    data = doc.tobytes()
    doc.close()
    return data
