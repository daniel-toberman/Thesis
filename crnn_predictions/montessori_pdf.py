from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -----------------------
# CONFIG
# -----------------------

FONT_PATH = "Comfortaa-Regular.ttf"
FONT_NAME = "Comfortaa"

PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE

LEFT_MARGIN = 30
RIGHT_MARGIN = 30
TOP_MARGIN = 30
BOTTOM_MARGIN = 30

ROWS_PER_PAGE = 6
BOX_HEIGHT = (PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) / ROWS_PER_PAGE
BOX_PADDING = 8

START_FONT_SIZE = 30
MIN_FONT_SIZE = 14

# -----------------------
# SENTENCES (100 unique)
# -----------------------

sentences = [
    "The cat and the dog ran together.",
    "My sister and I built a fort.",
    "Tom and Maya planted seeds outside.",
    "The wind and rain shook the trees.",
    "The teacher and students cleaned the room.",
    "Mom and Dad cooked dinner together.",
    "Birds and butterflies filled the garden.",
    "The sun and clouds shared the sky.",
    "My brother and his friend laughed loudly.",
    "The frog and turtle swam slowly.",

    "The cat ran and climbed the fence.",
    "She laughed and waved at us.",
    "The baby cried but then slept.",
    "He jumped and splashed in puddles.",
    "They ran and hid behind bushes.",
    "The dog barked and chased the ball.",
    "She opened the box and smiled.",
    "The wind blew and scattered leaves.",
    "We sang and danced in class.",
    "The boy slipped but stood quickly.",

    "The cat hid because it was scared.",
    "She whispered while the baby slept.",
    "They waited because the bus was late.",
    "He smiled although the rain fell.",
    "We stayed inside because it snowed.",
    "The dog barked while she knocked.",
    "She cried because her toy broke.",
    "They ran although the ground was wet.",
    "He listened while she told stories.",
    "We hurried because the bell rang.",

    "Wow, the stars shine brightly tonight.",
    "Oops, the milk spilled on the floor.",
    "Hey, that is my red ball.",
    "Hooray, we finished the puzzle.",
    "Yikes, the spider crawled near me.",
    "Ouch, I stepped on a rock.",
    "Oh dear, the birthday cake burned slightly.",
    "Wow, those birds fly very high.",
    "Oh my, the strong wind knocked down the oak tree.",

    "Oh no, the tall tower I built fell apart.",
    "Wow, the fireworks lit the sky.",
    "She and her friend shared snacks.",
    "The puppy and kitten slept together.",
    "He and I carried the heavy box carefully.",
    "The rain and wind returned suddenly.",
    "They smiled happily and waved at me.",
    "My cousins and I baked five cookies.",
    "The bell rang and we lined up.",
]


# -----------------------
# SETUP
# -----------------------

pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
c = canvas.Canvas("Montessori_Grammar_Sentence_Strips.pdf", pagesize=PAGE_SIZE)

usable_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

# -----------------------
# FUNCTIONS
# -----------------------

def fit_font_size(sentence, max_width):
    """Find the largest font size that fits the sentence on one line."""
    for size in range(START_FONT_SIZE, MIN_FONT_SIZE - 1, -1):
        text_width = pdfmetrics.stringWidth(sentence, FONT_NAME, size)
        if text_width <= max_width:
            return size
    return MIN_FONT_SIZE


def draw_justified_sentence(c, sentence, x, y, box_width, font_size):
    """Draws one fully justified line by stretching spaces."""
    words = sentence.split(" ")
    space_count = len(words) - 1

    c.setFont(FONT_NAME, font_size)

    if space_count == 0:
        c.drawString(x, y, sentence)
        return

    words_width = sum(
        pdfmetrics.stringWidth(word, FONT_NAME, font_size)
        for word in words
    )

    total_space = box_width - words_width
    space_width = total_space / space_count

    cursor_x = x
    for word in words:
        c.drawString(cursor_x, y, word)
        cursor_x += pdfmetrics.stringWidth(word, FONT_NAME, font_size)
        cursor_x += space_width


# -----------------------
# DRAW
# -----------------------

row_index = 0

for sentence in sentences:
    if row_index == ROWS_PER_PAGE:
        c.showPage()
        row_index = 0

    top_y = PAGE_HEIGHT - TOP_MARGIN - row_index * BOX_HEIGHT
    box_y = top_y - BOX_HEIGHT

    box_x = LEFT_MARGIN
    box_width = usable_width

    # Draw bounding box
    c.rect(box_x, box_y, box_width, BOX_HEIGHT, stroke=1, fill=0)

    # Fit font
    max_text_width = box_width - 2 * BOX_PADDING
    font_size = fit_font_size(sentence, max_text_width)

    text_y = box_y + (BOX_HEIGHT - font_size) / 2 - 2
    text_x = box_x + BOX_PADDING

    draw_justified_sentence(
        c,
        sentence,
        text_x,
        text_y,
        max_text_width,
        font_size
    )

    row_index += 1

c.save()
