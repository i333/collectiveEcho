#!/usr/bin/env python3
"""Render gallery_card.md to a one-page PDF using reportlab.

Hand-laid (not generic markdown→pdf) so the typography matches the tone
of the piece — generous margins, italic phrase headings, an em-rule
between sections instead of horizontal lines.
"""
import os
import sys

ROOT = "/home/utku/Desktop/projects/collectiveEcho"
SRC = os.path.join(ROOT, "gallery_card.md")
DEST = os.path.join(ROOT, "gallery_card.pdf")

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, KeepTogether, HRFlowable,
)
from reportlab.lib.colors import HexColor


# --- Styles -----------------------------------------------------------

INK = HexColor("#1a1a1a")
SOFT = HexColor("#5a5a5a")

styles = {
    "title": ParagraphStyle(
        "title", fontName="Times-Roman", fontSize=24, leading=28,
        alignment=TA_CENTER, textColor=INK, spaceAfter=4,
    ),
    "tagline": ParagraphStyle(
        "tagline", fontName="Times-Italic", fontSize=12, leading=14,
        alignment=TA_CENTER, textColor=SOFT, spaceAfter=18,
    ),
    "h2": ParagraphStyle(
        "h2", fontName="Times-Bold", fontSize=12, leading=15,
        alignment=TA_LEFT, textColor=INK,
        spaceBefore=10, spaceAfter=4,
    ),
    "phrase": ParagraphStyle(
        "phrase", fontName="Times-BoldItalic", fontSize=11.5, leading=14,
        alignment=TA_LEFT, textColor=INK,
        spaceBefore=8, spaceAfter=2,
    ),
    "body": ParagraphStyle(
        "body", fontName="Times-Roman", fontSize=10.5, leading=14,
        alignment=TA_LEFT, textColor=INK, spaceAfter=6,
    ),
    "list": ParagraphStyle(
        "list", fontName="Times-Roman", fontSize=10.5, leading=14,
        alignment=TA_LEFT, textColor=INK,
        leftIndent=16, bulletIndent=4, spaceAfter=2,
    ),
    "closing": ParagraphStyle(
        "closing", fontName="Times-Italic", fontSize=10.5, leading=14,
        alignment=TA_CENTER, textColor=SOFT, spaceBefore=14,
    ),
}


def hr(color=SOFT, thickness=0.5, width="60%"):
    return HRFlowable(width=width, thickness=thickness, color=color,
                       spaceBefore=8, spaceAfter=8, hAlign="CENTER")


def main():
    doc = SimpleDocTemplate(
        DEST,
        pagesize=LETTER,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="Collective Echo — Mercury Mine Monster",
        author="Collective Echo",
    )

    story = []

    # Title block
    story.append(Paragraph("Collective Echo", styles["title"]))
    story.append(Paragraph("Mercury Mine Monster", styles["tagline"]))
    story.append(hr())

    # Vision
    story.append(Paragraph("The Vision", styles["h2"]))
    story.append(Paragraph("A box that holds your voice.", styles["body"]))
    story.append(Paragraph(
        "Some things only land when said out loud — and even more so when "
        "echoed back through the voices of others who said them before you. "
        "This piece listens for what you say, weaves it into a growing "
        "chorus of everyone who has spoken into it tonight, and gives it "
        "back to you as a chord rather than a note.",
        styles["body"]))
    story.append(Paragraph(
        "What you put in stays in. Every echo from this moment forward "
        "carries a little of you in it.",
        styles["body"]))

    story.append(hr())

    # Sandy
    story.append(Paragraph("For Sandy", styles["h2"]))
    story.append(Paragraph(
        "This piece is dedicated to <b>Sandy</b> — a dear friend who is in "
        "the hospital tonight and couldn't be here in person. Her voice has "
        "been recorded for the show. Calling her by name lets her answer "
        "back, and lets us all be in the room with her, even from far away.",
        styles["body"]))

    story.append(hr())

    # How it works
    story.append(Paragraph("How It Works", styles["h2"]))
    story.append(Paragraph("The box is always listening.", styles["body"]))
    for i, line in enumerate([
        "Speak one of the phrases below near the mic.",
        "The lights flash to acknowledge that it heard you.",
        "Your voice is captured and mixed into the chorus of everyone who came before — older voices recede; the newest sit forward.",
        "The chorus plays back through the speaker, and the lights breathe along with it.",
        "When the lights settle back into a soft glow, the box is ready to hear you again.",
    ], 1):
        story.append(Paragraph(f"{i}. {line}", styles["list"]))

    story.append(hr())

    # Phrases
    story.append(Paragraph("What to Say", styles["h2"]))

    story.append(Paragraph("&ldquo;I am a monster&rdquo;", styles["phrase"]))
    story.append(Paragraph(
        "The heart of the piece. <b>Mercury Mine Monster</b> is about "
        "claiming the parts of ourselves we usually hide — and finding out, "
        "when we say it out loud, that we are not alone in having them. "
        "Every voice that says it joins a chorus of others who have done "
        "the same. There is room for all of it.",
        styles["body"]))

    story.append(Paragraph("&ldquo;I love you&rdquo;", styles["phrase"]))
    story.append(Paragraph(
        "An extension of the same gesture, in the other direction. If "
        "<i>I am a monster</i> is the courage to be seen, <i>I love you</i> "
        "is the courage to see someone else. Speak it however you want — "
        "whispered, called out, soft, sharp. The chorus will carry your "
        "tone forward.",
        styles["body"]))

    story.append(Paragraph("&ldquo;I love you, Sandy&rdquo;", styles["phrase"]))
    story.append(Paragraph(
        "Calls Sandy's voice from her own small archive. Only her voice "
        "answers; what you say is held privately and not mixed into anyone "
        "else's chorus. A way to send love directly to her tonight.",
        styles["body"]))

    story.append(Paragraph(
        "Take your time. Speak again if you'd like. The chorus grows with the night.",
        styles["closing"]))

    doc.build(story)
    print(f"Wrote {DEST}")
    print(f"Size: {os.path.getsize(DEST)} bytes")


if __name__ == "__main__":
    main()
