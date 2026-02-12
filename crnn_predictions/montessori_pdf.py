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

START_FONT_SIZE = 24
MIN_FONT_SIZE = 14

# -----------------------
# SENTENCES (100 unique)
# -----------------------

sentences = [
    "The curious fox explored the misty forest quietly.",
    "The tired child carried her heavy backpack home slowly.",
    "A small brown bird flew suddenly above the tall green trees.",
    "The artist painted bright colors on the wall with great care.",
    "Playful puppies chased the rolling ball eagerly before sunset.",
    "Loud thunder shook the old wooden house unexpectedly.",
    "The brave hiker stepped carefully over wet stones on the narrow path.",
    "The baby slept softly beside the warm window.",
    "Cheerful vendors called out fresh fruit prices loudly at the market.",
    "The shy cat watched the noisy children cautiously behind the fence.",
    "Shiny puddles reflected the pale evening sky after the rain.",
    "The swimmer entered the cold water calmly with a deep breath.",
    "Tall grasses waved gently in the cool breeze near the river.",
    "The tiny mouse nibbled a dry crumb nervously inside the box.",
    "The organized teacher arranged colorful papers neatly before school.",
    "The strong horse pulled the heavy cart proudly across the field.",
    "Bright stars filled the dark velvet sky quietly at night.",
    "The child discovered a forgotten toy happily with sudden joy.",
    "The playful dog waited patiently for crumbs under the table.",
    "The family shared stories peacefully by candlelight after dinner.",
    "Gentle waves touched the warm sand softly along the beach.",
    "Busy bees moved quickly between fragrant flowers in the garden.",
    "Nervous actors practiced their lines silently before the show.",
    "Gray clouds drifted slowly toward the hills above the city.",
    "Laughing children ran joyfully across the yard during recess.",
    "The thoughtful girl wrote a long letter carefully near the window.",
    "Dusty books waited quietly to be opened on the shelf.",
    "The tired athlete drank cool water gratefully after practice.",
    "The baker kneaded soft dough patiently with strong hands.",
    "Bright signs pointed clearly toward the town beside the road.",
    "Focused students listened attentively to instructions in the classroom.",
    "The gentle parent read a favorite story calmly before bedtime.",
    "A lonely boat drifted slowly without sound across the lake.",
    "The alert squirrel hid a small nut quickly on the branch.",
    "Busy hallways filled suddenly with voices after the bell.",
    "Warm soup simmered quietly on the stove in the kitchen.",
    "Tall mountains stood proudly against the wind under the sky.",
    "The dancer crossed the wide stage gracefully with careful steps.",
    "The old cat stretched its paws sleepily near the fire.",
    "Soft light touched the sleeping village gently at dawn.",
    "Quiet whispers floated through the dark air inside the tent.",
    "Curious questions filled the room eagerly after the lesson.",
    "Fallen leaves covered the ground softly along the trail.",
    "The mechanic fixed the loose bolt carefully with steady focus.",
    "Excited performers waited nervously behind the curtain.",
    "Tense runners tied their shoes silently before the race.",
    "Colorful kites climbed slowly into the sky in the yard.",
    "Muddy boots waited patiently to be cleaned near the door.",
    "Happy memories filled their thoughts warmly after the trip.",
    "Rushing water echoed loudly below the bridge.",
    "The student solved the problem confidently with quiet pride.",
    "Warm bread cooled patiently beside the soup at the table.",
    "Gentle shadows moved softly on the wall under the lamp.",
    "Golden wheat bent slowly in the wind across the field.",
    "Clean hands washed carefully in the sink before the meal.",
    "Green frogs jumped suddenly into the water near the pond.",
    "The crowd awaited the music eagerly with growing excitement.",
    "Lost keys rested quietly together inside the drawer.",
    "Brave laughter replaced surprised tears quickly after the fall.",
    "Tall trees watched passing cars calmly along the road.",
    "Raindrops tapped a gentle rhythm steadily at the window.",
    "The child learned to tie knots slowly with patient effort.",
    "Focused minds reviewed each answer carefully before the test.",
    "Sleepy animals chewed fresh hay quietly in the barn.",
    "Whispered wishes drifted upward softly under the stars.",
    "Tired legs welcomed rest happily after the climb.",
    "Fallen leaves gathered silently in piles near the bench.",
    "The nurse checked the patient gently with honest care.",
    "Neat letters formed clear words carefully across the page.",
    "Swinging chains creaked softly in the breeze at the park.",
    "The air held its breath strangely before the storm.",
    "Dripping water echoed through the darkness inside the cave.",
    "The toddler explored the room eagerly with curious eyes.",
    "Glowing light blinked briefly and faded near the firefly.",
    "Muddy shoes lined the porch quietly after the game.",
    "Climbing vines reached higher slowly along the fence.",
    "Warm colors spread gently across the sky at sunset.",
    "The gardener pulled tiny weeds carefully with calm patience.",
    "Packed bags waited neatly by the door before the trip.",
    "Small footprints marked the path quietly in the snow.",
    "Secret giggles escaped the dark softly under the blanket.",
    "Joyful applause filled the hall quickly after the song.",
    "Ticking seconds measured time steadily near the clock.",
    "The letter traveled far away slowly with quiet hope.",
    "Satisfied smiles shared success warmly at the end.",
    "Early in the morning, the curious fox quietly explored the misty forest.",
    "After lunch, the tired child slowly carried her heavy backpack home.",
    "The small brown bird suddenly flew above the tall green trees.",
    "With great care, the artist gently painted bright colors on the wall.",
    "Before sunset, the playful puppies eagerly chased the rolling ball.",
    "During the storm, loud thunder unexpectedly shook the old wooden house.",
    "On the narrow path, the brave hiker carefully stepped over wet stones.",
    "In the quiet room, the baby softly slept beside the warm window.",
    "At the market, cheerful vendors loudly called out fresh fruit prices.",
    "Behind the fence, the shy cat cautiously watched the noisy children.",
    "After the rain, shiny puddles reflected the pale evening sky.",
    "With a deep breath, the swimmer calmly entered the cold water.",
    "Near the river, tall grasses gently waved in the cool breeze.",
    "Inside the box, the tiny mouse nervously nibbled a dry crumb.",
    "Before school, the organized teacher neatly arranged colorful papers.",
    "Across the field, the strong horse proudly pulled the heavy cart.",
    "At night, bright stars quietly filled the dark velvet sky.",
    "With sudden joy, the child happily discovered a forgotten toy.",
    "Under the table, the playful dog patiently waited for crumbs.",
    "After dinner, the family peacefully shared stories by candlelight.",
    "Along the beach, gentle waves softly touched the warm sand.",
    "In the garden, busy bees quickly moved between fragrant flowers.",
    "Before the show, nervous actors silently practiced their lines.",
    "Above the city, gray clouds slowly drifted toward the hills.",
    "During recess, laughing children joyfully ran across the yard.",
    "Near the window, the thoughtful girl carefully wrote a long letter.",
    "On the shelf, dusty books quietly waited to be opened.",
    "After practice, the tired athlete gratefully drank cool water.",
    "With strong hands, the baker patiently kneaded soft dough.",
    "Beside the road, bright signs clearly pointed toward the town.",
    "In the classroom, focused students attentively listened to instructions.",
    "Before bedtime, the gentle parent calmly read a favorite story.",
    "Across the lake, a lonely boat slowly drifted without sound.",
    "On the branch, the alert squirrel quickly hid a small nut.",
    "After the bell, busy hallways suddenly filled with voices.",
    "In the kitchen, warm soup quietly simmered on the stove.",
    "Under the sky, tall mountains proudly stood against the wind.",
    "With careful steps, the dancer gracefully crossed the wide stage.",
    "Near the fire, the old cat sleepily stretched its paws.",
    "At dawn, soft light gently touched the sleeping village.",
    "Inside the tent, quiet whispers floated through the dark air.",
    "After the lesson, curious questions eagerly filled the room.",
    "Along the trail, fallen leaves softly covered the ground.",
    "With steady focus, the mechanic carefully fixed the loose bolt.",
    "Behind the curtain, excited performers nervously waited their turn.",
    "Before the race, tense runners silently tied their shoes.",
    "In the yard, colorful kites slowly climbed into the sky.",
    "Near the door, muddy boots patiently waited to be cleaned.",
    "After the trip, happy memories warmly filled their thoughts.",
    "Across the bridge, rushing water loudly echoed below.",
    "With quiet pride, the student confidently solved the problem.",
    "At the table, warm bread patiently cooled beside the soup.",
    "Under the lamp, gentle shadows softly moved on the wall.",
    "In the field, golden wheat slowly bent in the wind.",
    "Before the meal, clean hands carefully washed in the sink.",
    "Near the pond, green frogs suddenly jumped into the water.",
    "With growing excitement, the crowd eagerly awaited the music.",
    "Inside the drawer, lost keys quietly rested together.",
    "After the fall, brave laughter quickly replaced surprised tears.",
    "Along the road, tall trees calmly watched passing cars.",
    "At the window, raindrops steadily tapped a gentle rhythm.",
    "With patient effort, the child slowly learned to tie knots.",
    "Before the test, focused minds carefully reviewed each answer.",
    "In the barn, sleepy animals quietly chewed fresh hay.",
    "Under the stars, whispered wishes softly drifted upward.",
    "After the climb, tired legs happily welcomed rest.",
    "Near the bench, fallen leaves silently gathered in piles.",
    "With honest care, the nurse gently checked the patient.",
    "Across the page, neat letters carefully formed clear words.",
    "At the park, swinging chains softly creaked in the breeze.",
    "Before the storm, the air strangely held its breath.",
    "Inside the cave, dripping water echoed through the darkness.",
    "With curious eyes, the toddler eagerly explored the room.",
    "Near the firefly, glowing light briefly blinked and faded.",
    "After the game, muddy shoes quietly lined the porch.",
    "Along the fence, climbing vines slowly reached higher.",
    "At sunset, warm colors gently spread across the sky.",
    "With calm patience, the gardener carefully pulled tiny weeds.",
    "Before the trip, packed bags neatly waited by the door.",
    "In the snow, small footprints quietly marked the path.",
    "Under the blanket, secret giggles softly escaped the dark.",
    "After the song, joyful applause quickly filled the hall.",
    "Near the clock, ticking seconds steadily measured time.",
    "With quiet hope, the letter slowly traveled far away.",
    "At the end, satisfied smiles warmly shared success."
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
