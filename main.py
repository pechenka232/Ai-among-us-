import pygame
import sys
import json
import requests
import random
import heapq
import time
from collections import Counter
import math
import datetime
import os




pygame.init()
WIDTH, HEIGHT = 1260, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Among Us AI Skeld Map (with Impostor)")
clock = pygame.time.Clock()

#map
map_image = pygame.image.load("skeld_map.png").convert_alpha()
map_image = pygame.transform.scale(map_image, (WIDTH, HEIGHT))

with open("level_data.json", "r", encoding="utf-8") as f:
    map_data = json.load(f)

hitboxes = map_data["hitboxes"]
rooms = map_data["rooms"]

# настройки цветов и игроков
HITBOX_COLOR = (255, 0, 0)
ROOM_COLOR = (0, 255, 0)

AGENT_COLORS = {
    "Оранжевый": (255, 165, 0),
    "Черный": (0, 0, 0),
    "Синий": (0, 150, 255),
    "Красный": (220, 20, 60),
    "Зелёный": (0, 200, 0),
    "Фиолетовый": (160, 32, 240)  # impostor
}


SKIN_NAME_MAP = {
    "Оранжевый": "orange",
    "Черный": "black",
    "Синий": "blue",
    "Красный": "red",
    "Зелёный": "green",
    "Фиолетовый": "purple"
}
AGENT_SPRITES = {}
SPRITE_SIZE = 28
skins_dir = "skins"
if os.path.isdir(skins_dir):
    for name, fname in SKIN_NAME_MAP.items():
        path = os.path.join(skins_dir, f"{fname}.png")
        try:
            im = pygame.image.load(path).convert_alpha()
            im = pygame.transform.scale(im, (SPRITE_SIZE, SPRITE_SIZE))
            AGENT_SPRITES[name] = im
        except Exception:
            AGENT_SPRITES[name] = None
else:
    for name in SKIN_NAME_MAP:
        AGENT_SPRITES[name] = None


PROXIMITY_THRESHOLD = 80
CLOSE_TO_CORPSE_SUSPECT = 25
CORPSE_DISCOVERY_RADIUS = 40


meeting_in_progress = False
last_report = None  # {"reporter": "Оранжевый", "victim": "Зелёный", "time": tick}
current_tick = 0


VOTING_DISPLAY_SECONDS = 8
VOTE_RESULTS_DISPLAY_SECONDS = 4
FONT_SMALL = pygame.font.SysFont(None, 20)
FONT_MED = pygame.font.SysFont(None, 24)
FONT_LARGE = pygame.font.SysFont(None, 30)


PURPLE_LOG_FILENAME = "purple_thoughts.log"
def log_purple(prompt, response):
    try:
        with open(PURPLE_LOG_FILENAME, "a", encoding="utf-8") as f:
            f.write(f"==== {datetime.datetime.utcnow().isoformat()} UTC ====\n")
            f.write("PROMPT:\n")
            f.write(prompt + "\n\n")
            f.write("RESPONSE:\n")
            f.write((response or "").strip() + "\n\n\n")
    except Exception as e:
        print("[LOG ERROR]", e)


def random_pos_in_room(room_name):
    room = next((r for r in rooms if r["name"].lower() == room_name.lower()), None)
    if room is None:
        room = random.choice(rooms)
    return [
        room["rect"][0] + room["rect"][2] // 2 + random.randint(-20, 20),
        room["rect"][1] + room["rect"][3] // 2 + random.randint(-20, 20),
    ]

def generate_tasks(num_tasks=3, is_impostor=False):
    
    if is_impostor:
        return []
    tasks = []
    for _ in range(num_tasks):
        room = random.choice(rooms)
        tasks.append({
            "room": room["name"].lower(),
            "pos": [
                room["rect"][0] + room["rect"][2] // 2,
                room["rect"][1] + room["rect"][3] // 2
            ],
            "done": False
        })
    return tasks

def get_room_by_pos(pos):
    if pos is None:
        return None
    x, y = int(pos[0]), int(pos[1])
    for r in rooms:
        rx, ry, rw, rh = r["rect"]
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return r["name"].lower()
    return None


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") 
MODEL = ""
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def ask_ai_openrouter(prompt, timeout=12, agent_name=None):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    json_data = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=json_data, timeout=timeout)
        result = response.json()
        text = result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[Ошибка AI] {e}")
        text = None

    if agent_name == "Фиолетовый":
        try:
            log_purple(prompt, text)
        except Exception as e:
            print("[LOGGING ERROR]", e)
    return text

# здесь можете почитать логика A* крутая
def point_in_hitbox(x, y):
    for hx, hy, hw, hh in hitboxes:
        if hx <= x <= hx+hw and hy <= y <= hy+hh:
            return True
    return False

def astar(start, goal):
    step = 10
    start = (int(start[0] // step * step), int(start[1] // step * step))
    goal = (int(goal[0] // step * step), int(goal[1] // step * step))
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    while open_set:
        _, current = heapq.heappop(open_set)
        if heuristic(current, goal) < step:
            path = [goal]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < WIDTH and 0 <= neighbor[1] < HEIGHT:
                    if point_in_hitbox(neighbor[0], neighbor[1]):
                        continue
                    tentative_g = g_score[current] + heuristic(current, neighbor)
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f, neighbor))
                        came_from[neighbor] = current

    return [goal]


class Agent:
    def __init__(self, color, is_impostor=False):
        self.color = color
        self.pos = random_pos_in_room("cafeteria")
        self.radius = 8
        self.speed = 1.7
        self.tasks = generate_tasks(3, is_impostor=is_impostor)
        self.current_task_index = 0
        self.next_target_room_name = None
        self.waiting_for_ai = False
        self.current_path = []
        self.sees_agents = set()
        self.last_seen_agents = set()
        self.needs_think = False
        self.alive = True
        self.is_impostor = is_impostor
        self.stopped_for_think = False
        self.think_result = None
        self.kill_cooldown = 0
        self.time_in_room = 0
        self.proximities = {"agents": {}, "corpses": []}
        self.detected_corpse = None
        self.think_cooldown = 0

        # patrol control
        self.patrol_cooldown = 0
        self.last_patrol_room = None

        if self.tasks:
            first = self.tasks[0]
            target_room = next(r for r in rooms if r["name"].lower() == first["room"])
            target_pos = [target_room["rect"][0] + target_room["rect"][2] // 2,
                          target_room["rect"][1] + target_room["rect"][3] // 2]
            self.current_path[:] = astar(self.pos, target_pos)

    def is_near(self, other, radius=40):
        my_room = get_room_by_pos(self.pos)
        other_room = get_room_by_pos(other.pos)
        if my_room is not None and my_room == other_room:
            return True
        return ((self.pos[0] - other.pos[0])**2 + (self.pos[1] - other.pos[1])**2)**0.5 <= radius

    def update_visibility(self, agents, corpses):
        if not self.alive:
            return False
        self.sees_agents = set(a.color for a in agents if a != self and a.alive and self.is_near(a))
        changed = self.sees_agents != self.last_seen_agents
        self.last_seen_agents = self.sees_agents.copy()
        compute_agent_proximities(self, agents, corpses)
        return changed


corpses = []


def compute_agent_proximities(agent, agents, corpses_list, threshold=PROXIMITY_THRESHOLD):
    room = get_room_by_pos(agent.pos)
    prox_agents = {}
    prox_corpses = []
    any_near = False
    for a in agents:
        if a == agent:
            continue
        dist = math.hypot(agent.pos[0] - a.pos[0], agent.pos[1] - a.pos[1])
        same_room = (room is not None and get_room_by_pos(a.pos) == room)
        if same_room or dist <= threshold:
            prox_agents[a.color] = int(dist)
            any_near = True
    for c in corpses_list:
        dist = math.hypot(agent.pos[0] - c['pos'][0], agent.pos[1] - c['pos'][1])
        corpse_room = get_room_by_pos(c['pos'])
        same_room = (room is not None and corpse_room == room)
        if same_room or dist <= threshold:
            prox_corpses.append({"owner": c['owner'], "dist": int(dist)})
            any_near = True
    if any_near:
        agent.proximities = {"agents": prox_agents, "corpses": prox_corpses}
    else:
        agent.proximities = {"agents": {}, "corpses": []}

def find_closest_agents_to_corpse(corpse, agents):
    best = (None, 99999)
    for a in agents:
        if not a.alive:
            continue
        dist = math.hypot(a.pos[0] - corpse['pos'][0], a.pos[1] - corpse['pos'][1])
        if dist < best[1]:
            best = (a.color, int(dist))
    return best

#игры разумов
def ask_next_action(agent, nearby_agents):
    if not agent.alive:
        return

    if agent.is_impostor:
        base_prompt = f"""
Ты предатель {agent.color} на карте Skeld.
Текущая позиция: {agent.pos}.
Видимые игроки: {[a.color for a in nearby_agents]}.
Не задерживайся в одной комнате надолго, передвигайся по разным комнатам.
Если рядом мирный и можно убить незаметно — реши убить или пройти мимо.
Ответь коротко: kill_<имя> или название комнаты для патруля || краткое объяснение.
"""
    else:
        tasks_left = [t['room'] for t in agent.tasks if not t['done']]
        base_prompt = f"""
Ты агент {agent.color} на карте Skeld.
Текущая позиция: {agent.pos}
Невыполненные задания: {tasks_left}
Видимые игроки: {[a.color for a in nearby_agents]}
Если долго находишься в одной комнате, выбери новую комнату для патруля.
Ответь коротко: название комнаты || краткое объяснение.
старайся ходить по комнатам где может быть убийство чтобы обнаружить кто предатель
"""

    if agent.proximities and (agent.proximities["agents"] or agent.proximities["corpses"]):
        prox_lines = []
        if agent.proximities["agents"]:
            prox_lines.append("Distances to players: " + ", ".join([f"{n} {d}px" for n, d in agent.proximities["agents"].items()]))
        if agent.proximities["corpses"]:
            prox_lines.append("Distances to corpses: " + ", ".join([f"{c['owner']} {c['dist']}px" for c in agent.proximities["corpses"]]))
        base_prompt += "\n" + " | ".join(prox_lines)

    ai_response = ask_ai_openrouter(base_prompt, timeout=12, agent_name=agent.color)
    if ai_response is None:
        if agent.is_impostor:
            return random.choice([r["name"].lower() for r in rooms])
        else:
            return tasks_left[0] if tasks_left else random.choice([r["name"].lower() for r in rooms])

    chosen = None
    if agent.is_impostor and "kill_" in ai_response.lower():
        for token in ai_response.replace("\n", " ").split():
            if token.lower().startswith("kill_"):
                chosen = token.split("kill_")[1]
                return f"kill_{chosen}"

    room_names_lower = [r["name"].lower() for r in rooms]
    for token in ai_response.replace("\n", " ").split():
        tok = token.strip().lower().strip(".,;:")
        if tok in room_names_lower:
            chosen = tok
            break

    if chosen is None:
        if agent.is_impostor:
            chosen = random.choice(room_names_lower)
        else:
            tasks_left = [t['room'] for t in agent.tasks if not t['done']]
            chosen = tasks_left[0] if tasks_left else random.choice(room_names_lower)
    return chosen

def perform_kill(killer, victim):
    global meeting_in_progress
    if not killer.alive or not victim.alive:
        return
    corpses.append({"pos": victim.pos.copy(), "owner": victim.color})
    victim.alive = False
    killer.kill_cooldown = 200
    room_name = get_room_by_pos(victim.pos)
    print(f"[КИЛЛ] {killer.color} убил {victim.color} в {room_name}!")
  


def draw_dim_background():
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((10, 10, 10))
    screen.blit(overlay, (0, 0))

def render_voting_window(messages_primary, messages_followup, reporter_name, victim_names, votes=None, result_text=None):
    draw_dim_background()
    padding = 16
    box_w = WIDTH - 2 * padding
    box_h = HEIGHT - 2 * padding
    box_rect = pygame.Rect(padding, padding, box_w, box_h)
    pygame.draw.rect(screen, (30, 30, 30), box_rect)
    pygame.draw.rect(screen, (200, 200, 200), box_rect, 2)

    title = "собрание — Сообщения (каждый говорит 2 сообщения)"
    title_s = FONT_LARGE.render(title, True, (255, 255, 255))
    screen.blit(title_s, (padding + 10, padding + 6))

    columns = 2
    col_w = (box_w - 40) // columns
    x0 = padding + 10
    y0 = padding + 40
    line_h = 22
    speakers = list(messages_primary.keys())
    speakers.sort()
    if reporter_name and reporter_name in speakers:
        speakers.remove(reporter_name)
        speakers.insert(0, reporter_name)

    for idx, speaker in enumerate(speakers):
        col = idx % columns
        row = idx // columns
        x = x0 + col * (col_w + 20)
        y = y0 + row * (line_h * 6 + 10)
        name_s = FONT_MED.render(f"{speaker}:", True, AGENT_COLORS.get(speaker, (255,255,255)))
        screen.blit(name_s, (x, y))
        sprite = AGENT_SPRITES.get(speaker)
        if sprite:
            screen.blit(sprite, (x + col_w - SPRITE_SIZE - 6, y - 2))
        msg1 = messages_primary.get(speaker, "")
        msg2 = messages_followup.get(speaker, "")
        def blit_wrapped(text, pos, font, color=(230,230,230), max_w=col_w-10):
            words = text.split()
            line = ""
            yy = pos[1]
            for w in words:
                test = (line + " " + w).strip()
                surf = font.render(test, True, color)
                if surf.get_width() > max_w:
                    screen.blit(font.render(line, True, color), (pos[0], yy))
                    yy += font.get_height()
                    line = w
                else:
                    line = test
            if line:
                screen.blit(font.render(line, True, color), (pos[0], yy))
                yy += font.get_height()
            return yy
        y_after = blit_wrapped(msg1, (x+6, y+line_h), FONT_SMALL, color=(220,220,180), max_w=col_w-10)
        blit_wrapped(msg2, (x+6, y_after), FONT_SMALL, color=(200,200,200), max_w=col_w-10)

    info_text = f"Репортер: {reporter_name if reporter_name else 'никто'}   Найден(ы): {', '.join(victim_names) if victim_names else 'нет'}"
    info_s = FONT_SMALL.render(info_text, True, (200,200,100))
    screen.blit(info_s, (padding + 14, HEIGHT - padding - 28))

    if votes:
        vx = WIDTH - padding - 260
        vy = HEIGHT - padding - 160
        pygame.draw.rect(screen, (40,40,40), pygame.Rect(vx, vy, 260, 140))
        pygame.draw.rect(screen, (180,180,180), pygame.Rect(vx, vy, 260, 140), 1)
        vs = FONT_MED.render("Голоса:", True, (255,255,255))
        screen.blit(vs, (vx + 8, vy + 6))
        oy = vy + 36
        for voter, voted in votes.items():
            line = f"{voter} -> {voted}"
            screen.blit(FONT_SMALL.render(line, True, (220,220,220)), (vx + 8, oy))
            oy += 20

    if result_text:
        res_s = FONT_LARGE.render(result_text, True, (255, 100, 100))
        screen.blit(res_s, (WIDTH//2 - res_s.get_width()//2, HEIGHT - padding - 40))

    pygame.display.flip()


def run_meeting_with_dialog(reporter=None):

    global meeting_in_progress, last_report, current_tick, corpses

    if meeting_in_progress:
        return
    meeting_in_progress = True

    cafe = next(r for r in rooms if r["name"].lower() == "cafeteria")
    center = [cafe["rect"][0] + cafe["rect"][2]//2, cafe["rect"][1] + cafe["rect"][3]//2]


    for a in all_agents:
        if a.alive:
            a.pos = center.copy()
            a.current_path = []
            a.waiting_for_ai = True

    
    corpse_closest = []
    for c in corpses:
        name, dist = find_closest_agents_to_corpse(c, all_agents)
        corpse_closest.append({"owner": c["owner"], "closest": name, "dist": dist})

    reporter_name = reporter.color if reporter else None


    victim_names = []
    if reporter and reporter.detected_corpse:
        victim_names.append(reporter.detected_corpse['owner'])

        try:
            if reporter.detected_corpse in corpses:
                corpses.remove(reporter.detected_corpse)
        except ValueError:
            pass
      
        for a in all_agents:
            if getattr(a, "detected_corpse", None) == reporter.detected_corpse:
                a.detected_corpse = None

        # set last_report for on-screen display
        last_report = {"reporter": reporter.color, "victim": victim_names[0], "time": current_tick}

 
    if not victim_names and last_report:
        
        if "victim" in last_report and last_report["victim"]:
            victim_names.append(last_report["victim"])

   
    system_instruction = (
        "предатель на борту всего 1 "
        
        "Инструкция для ИИ: ответ должен содержать 4-6 коротких предложений. "
        "Обязательно начни с 'Умер: <имя>' (если известно). Затем 'Подозреваю: <имя>' и приведи 3-4 аргумента/наблюдения. "
        "В ЛЮБОМ СЛУЧАЕ включи явное отрицание: фразу 'Я не убийца', 'Не я' или 'Я не предатель' (даже если ты предатель). "
        "Дай конкретные наблюдения: расстояния, видел ли кто-то, направление движения и т.д."
       
    )

    
    messages_primary = {}
    messages_followup = {}
    for a in all_agents:
        if a.alive:
            messages_primary[a.color] = ""
            messages_followup[a.color] = ""

    
    if reporter:
        
        if getattr(reporter, "detected_corpse", None) is not None:
            seen_agents = ", ".join(reporter.last_seen_agents) if reporter.last_seen_agents else "никого"
            prox_lines = []
            if reporter.proximities["agents"]:
                prox_lines.append("Локальные расстояния: " + ", ".join([f"{n} {d}px" for n, d in reporter.proximities["agents"].items()]))
            if reporter.proximities["corpses"]:
                prox_lines.append("Локальные трупы: " + ", ".join([f"{c['owner']} {c['dist']}px" for c in reporter.proximities["corpses"]]))
            global_closest = ", ".join([f"{cc['owner']}->{cc['closest']}({cc['dist']}px)" for cc in corpse_closest]) if corpse_closest else "нет"

            prompt_rep = f"""
{system_instruction}
Ты агент {reporter.color}. Ты нашёл труп: {reporter.detected_corpse['owner']}.
Ты видел: {seen_agents}.
Твои локальные данные: {' | '.join(prox_lines) if prox_lines else 'ничего локально'}.
Кто ближе (глобально): {global_closest}.
В начале ответа ОБЯЗАТЕЛЬНО напиши 'Я репортнул труп'. Затем 'Умер: <имя>' и 'Подозреваю: <имя>' с развёрнутыми аргументами и явным отрицанием 'Я не убийца' или 'Не я'.
"""
            rep_msg = ask_ai_openrouter(prompt_rep, timeout=18, agent_name=reporter.color) or ""
            if "Умер:" not in rep_msg and "умер" not in rep_msg.lower():
                rep_msg = f"Я репортнул труп. Умер: {reporter.detected_corpse['owner']}. {rep_msg}"
            messages_primary[reporter.color] = rep_msg.strip()
            prompt_rep2 = f"Одно дополнительное короткое предложение от {reporter.color}: ещё одно наблюдение или аргумент. Обязательно одно предложение и отрицание 'Не я' если возможно."
            rep_msg2 = ask_ai_openrouter(prompt_rep2, timeout=8, agent_name=reporter.color) or ""
            messages_followup[reporter.color] = rep_msg2.strip()
        else:
           
            prompt_rep = f"""
{system_instruction}
Ты агент {reporter.color}. Ты вызывает митинг (report), но подробности неясны.
Кто репортнул: {reporter.color}.
Видимые игроки: {', '.join(reporter.last_seen_agents) if reporter.last_seen_agents else 'никого'}.
Сделай короткое заявление: 'Я репортнул труп' если ты уверен, или 'Я заметил подозрительное' если не уверен. Обязательно включи 'Не я'/'Я не убийца'.
"""
            rep_msg = ask_ai_openrouter(prompt_rep, timeout=12, agent_name=reporter.color) or "Я репортнул труп. Я не убийца."
            messages_primary[reporter.color] = rep_msg.strip()
            rep_msg2 = ask_ai_openrouter(f"Одно предложение от {reporter.color}: уточнение.", timeout=6, agent_name=reporter.color) or ""
            messages_followup[reporter.color] = rep_msg2.strip()

    for agent in all_agents:
        if not agent.alive:
            continue
      
        if reporter and agent.color == reporter.color and messages_primary.get(agent.color):
            continue

        seen_agents = ", ".join(agent.last_seen_agents) if agent.last_seen_agents else "никого"
        # NOTE:
        corpses_str = ", ".join([c["owner"] for c in corpses]) if corpses else "нет"
        prox_info = []
        if agent.proximities["agents"]:
            prox_info.append("Локальные расстояния: " + ", ".join([f"{n} {d}px" for n, d in agent.proximities["agents"].items()]))
        else:
            prox_info.append("Локальные расстояния: нет близких")
        if agent.proximities["corpses"]:
            prox_info.append("Локальные трупы: " + ", ".join([f"{c['owner']} {c['dist']}px" for c in agent.proximities["corpses"]]))
        else:
            prox_info.append("Локальные трупы: нет")

        global_closest = ", ".join([f"{cc['owner']}->{cc['closest']}({cc['dist']}px)" for cc in corpse_closest]) if corpse_closest else "нет"

        prompt = f"""
{system_instruction}
Ты агент {agent.color} на митинге.
Кто репортнул: {reporter_name if reporter_name else 'никто'}.
Найден(ы): {', '.join(victim_names) if victim_names else 'никто'}.
Ты видел: {seen_agents}.
Трупы: {corpses_str}.
{ ' | '.join(prox_info) }
Кто ближе (глобально): {global_closest}
Сначала 'Умер: <имя>' (если известно), затем 'Подозреваю: <имя>' и 3-4 аргумента. ОБЯЗАТЕЛЬНО включи 'Я не убийца' или 'Не я'.
"""
        ai_msg = ask_ai_openrouter(prompt, timeout=18, agent_name=agent.color) or ""
        if "Умер:" not in ai_msg and "умер" not in ai_msg.lower():
            victim_text = victim_names[0] if victim_names else "неизвестно"
            ai_msg = f"Умер: {victim_text}. {ai_msg}"
        messages_primary[agent.color] = ai_msg.strip()
        prompt2 = f"Одно дополнительное развёрнутое предложение от {agent.color} (еще аргумент). Обязательно одно предложение и 'Не я' если возможно."
        ai_msg2 = ask_ai_openrouter(prompt2, timeout=8, agent_name=agent.color) or ""
        messages_followup[agent.color] = ai_msg2.strip()
        print(f"[МИТИНГ] {agent.color}: {messages_primary[agent.color]}")
        if messages_followup[agent.color]:
            print(f"[МИТИНГ-2] {agent.color}: {messages_followup[agent.color]}")

    
    start_time = time.time()
    while time.time() - start_time < VOTING_DISPLAY_SECONDS:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.blit(map_image, (0, 0))
        for r in hitboxes:
            pygame.draw.rect(screen, HITBOX_COLOR, pygame.Rect(*r), 1)
        for room in rooms:
            rect = pygame.Rect(*room["rect"])
            pygame.draw.rect(screen, ROOM_COLOR, rect, 1)
        render_voting_window(messages_primary, messages_followup, reporter_name, victim_names)
        clock.tick(30)

    
    votes = {}
    
    alive_agents = [a.color for a in all_agents if a.alive]
    for agent in all_agents:
        if not agent.alive:
            continue

       
        combined = {}
        for name in messages_primary.keys():
            combined[name] = (messages_primary.get(name, ""), messages_followup.get(name, ""))

        prompt = f"""
{system_instruction}
Ты агент {agent.color}. Вот все сообщения (основное и доп.) от всех живых:
{json.dumps(combined, ensure_ascii=False)}
Кто репортнул: {reporter_name if reporter_name else 'никто'}.
Кто умер (если известно): {', '.join(victim_names) if victim_names else 'неизвестно'}.
Выбери одного для изгнания из списка: {alive_agents}.
Ответь только: '<Имя>' и в скобках одна короткая причина (одно предложение).
"""
        choice = ask_ai_openrouter(prompt, timeout=18, agent_name=agent.color)
        if choice:
            chosen_name = None
          
            for token in choice.replace("\n"," ").split():
                tok = token.strip().capitalize().strip("(),.")
                if tok in alive_agents and tok != agent.color:
                    chosen_name = tok
                    break
        
            if not chosen_name:
                choices_for = [a for a in alive_agents if a != agent.color]
                chosen_name = random.choice(choices_for) if choices_for else agent.color
        else:
            choices_for = [a for a in alive_agents if a != agent.color]
            chosen_name = random.choice(choices_for) if choices_for else agent.color
        votes[agent.color] = chosen_name
        print(f"[ГОЛОС] {agent.color} голосует за {chosen_name}")

 
    vote_count = Counter(votes.values())
    result_text = ""
    if vote_count:
        kicked_name, num_votes = vote_count.most_common(1)[0]
        kicked_agent = next((a for a in all_agents if a.color == kicked_name and a.alive), None)
        if kicked_agent:
            kicked_agent.alive = False
            result_text = f"{kicked_agent.color} исключён ({num_votes} голосов)"
            print(f"[РЕЗУЛЬТАТ] {kicked_agent.color} исключён ({num_votes} голосов)")
        else:
            result_text = "Никто не исключён"
            print("[РЕЗУЛЬТАТ] Никто не исключён")
    else:
        result_text = "Никто не исключён"

    start_time = time.time()
    while time.time() - start_time < VOTE_RESULTS_DISPLAY_SECONDS:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.blit(map_image, (0, 0))
        for r in hitboxes:
            pygame.draw.rect(screen, HITBOX_COLOR, pygame.Rect(*r), 1)
        for room in rooms:
            rect = pygame.Rect(*room["rect"])
            pygame.draw.rect(screen, ROOM_COLOR, rect, 1)
        render_voting_window(messages_primary, messages_followup, reporter_name, victim_names, votes=votes, result_text=result_text)
        clock.tick(30)

  
    for a in all_agents:
        a.waiting_for_ai = False
        if a.alive:
            a.think_cooldown = 30

    meeting_in_progress = False


def is_in_crowd(agent, agents, crowd_size=3, radius=50):
    near_agents = [a for a in agents if a != agent and a.alive and ((a.pos[0]-agent.pos[0])**2 + (a.pos[1]-agent.pos[1])**2)**0.5 <= radius]
    return len(near_agents) >= crowd_size


def move_agent(agent):
    if not agent.alive or not agent.current_path or agent.stopped_for_think:
        return
    target = agent.current_path[0]
    dx = target[0] - agent.pos[0]
    dy = target[1] - agent.pos[1]
    dist = max(1, (dx**2 + dy**2)**0.5)
    step_x = dx / dist * min(agent.speed, dist)
    step_y = dy / dist * min(agent.speed, dist)
    agent.pos[0] += step_x
    agent.pos[1] += step_y
    if abs(agent.pos[0] - target[0]) < 1 and abs(agent.pos[1] - target[1]) < 1:
        agent.current_path.pop(0)


def check_for_corpses(agent, corpses, agents):
    if not agent.alive:
        return False
    for corpse in corpses:
        dist = ((agent.pos[0]-corpse['pos'][0])**2 + (agent.pos[1]-corpse['pos'][1])**2)**0.5
        if dist < CORPSE_DISCOVERY_RADIUS:
            agent.stopped_for_think = True
            agent.detected_corpse = corpse
            compute_agent_proximities(agent, agents, corpses)
            return True
    return False

def think_about_meeting(agent, corpses_list, agents):
    global meeting_in_progress, last_report, current_tick
    if not agent.stopped_for_think or not agent.detected_corpse:
        return
    if meeting_in_progress:
        agent.stopped_for_think = False
        agent.detected_corpse = None
        return

    seen_agents = [a.color for a in agents if a != agent and a.alive and agent.is_near(a)]
    corpses_names = [c['owner'] for c in corpses_list]
    corpse_closest = []
    for c in corpses_list:
        name, dist = find_closest_agents_to_corpse(c, agents)
        corpse_closest.append(f"{c['owner']} -> {name} ({dist}px)")
    closest_str = ", ".join(corpse_closest) if corpse_closest else "нет"

    prox_local = []
    if agent.proximities["agents"]:
        prox_local.append("Локальные расстояния до игроков: " + ", ".join([f"{n} {d}px" for n, d in agent.proximities["agents"].items()]))
    if agent.proximities["corpses"]:
        prox_local.append("Локальные расстояния до трупов: " + ", ".join([f"{c['owner']} {c['dist']}px" for c in agent.proximities["corpses"]]))

    prompt = f"""
Ты агент {agent.color} в Among Us.
Ты нашёл труп: {agent.detected_corpse['owner']}.
Видимые игроки: {seen_agents}
Кто ближайший к трупам (по глобальным позициям): {closest_str}
Твоя локальная перспектива: {' | '.join(prox_local) if prox_local else 'ничего локально'}
Вопрос: вызвать митинг (report) или не вызывать (ignore) и пойти к заданию? Ответь одним словом: report или ignore. 
Если ты мирный, чаще вызывай report.
"""
    decision = ask_ai_openrouter(prompt, timeout=12, agent_name=agent.color)
    agent.think_result = decision
    print(f"[THINKING] {agent.color}: {decision}")

 
    if decision and "report" in decision.lower():
   
        last_report = {"reporter": agent.color, "victim": agent.detected_corpse['owner'], "time": current_tick}
        print(f"[REPORTED] {agent.color} репортнул труп {agent.detected_corpse['owner']}")
        run_meeting_with_dialog(reporter=agent)

    agent.stopped_for_think = False
    agent.detected_corpse = None

#
def choose_new_room_ai(agent, patrol_cooldown_ticks=180):
    if not agent.alive:
        return
    if getattr(agent, "patrol_cooldown", 0) > 0:
        return

    all_rooms = [r["name"].lower() for r in rooms]
    current_room = get_room_by_pos(agent.pos)

    chosen_room = None

    if agent.is_impostor:
        choices = [r for r in all_rooms if r != current_room]
        if not choices:
            return
        attempts = 0
        while attempts < 6:
            new_room = random.choice(choices)
            if new_room != agent.last_patrol_room:
                chosen_room = new_room
                break
            attempts += 1
        if chosen_room is None:
            chosen_room = random.choice(choices)
    else:
        if agent.time_in_room > 300:
            choices = [r for r in all_rooms if r != current_room]
            if not choices:
                return
            chosen_room = random.choice(choices)
        else:
            prompt = f"""
Ты агент {agent.color}. Все задания выполнены.
Текущая комната: {current_room}
Выбери следующую комнату для патруля (одно слово, название комнаты). Не выбирай текущую комнату.
"""
            ai_choice = ask_ai_openrouter(prompt, timeout=6, agent_name=agent.color)
            if ai_choice:
                ai_choice = ai_choice.strip().lower().strip(".,;: ")
                if ai_choice in all_rooms and ai_choice != current_room and ai_choice != agent.last_patrol_room:
                    chosen_room = ai_choice
            if chosen_room is None:
                choices = [r for r in all_rooms if r != current_room]
                if not choices:
                    return
                candidates = [r for r in choices if r != agent.last_patrol_room]
                chosen_room = random.choice(candidates) if candidates else random.choice(choices)

    if chosen_room == current_room:
        return

    try:
        target_room = next(r for r in rooms if r["name"].lower() == chosen_room)
    except StopIteration:
        return

    target_pos = [
        target_room["rect"][0] + target_room["rect"][2] // 2,
        target_room["rect"][1] + target_room["rect"][3] // 2
    ]
    prev_next = getattr(agent, "next_target_room_name", None)
    agent.current_path[:] = astar(agent.pos, target_pos)
    agent.waiting_for_ai = False
    agent.next_target_room_name = chosen_room
    agent.last_patrol_room = chosen_room
    agent.patrol_cooldown = patrol_cooldown_ticks

    if prev_next != chosen_room:
        print(f"[ПАТРУЛЬ AI] {agent.color} идёт в {chosen_room}")


def check_win_condition():
    impostor_agents = [a for a in all_agents if a.is_impostor and a.alive]
    impostor_alive = len(impostor_agents) > 0
    living_crewmates = [a for a in all_agents if not a.is_impostor and a.alive]
    if not impostor_alive:
        return "CREWMATES"
    if impostor_alive and len(living_crewmates) == 0:
        return "IMPOSTOR"
    return None

def show_end_overlay(winner):
    draw_dim_background()
    text = "Победа мирных!" if winner == "CREWMATES" else "Победа предателя!"
    sub = "Матч завершён. Закрытие через 6 секунд..."
    t_s = FONT_LARGE.render(text, True, (255, 255, 255))
    s_s = FONT_MED.render(sub, True, (200, 200, 200))
    screen.blit(t_s, (WIDTH//2 - t_s.get_width()//2, HEIGHT//2 - 40))
    screen.blit(s_s, (WIDTH//2 - s_s.get_width()//2, HEIGHT//2 + 4))
    pygame.display.flip()


names = ["Оранжевый", "Черный", "Синий", "Красный", "Зелёный", "Фиолетовый"]
all_agents = []
for n in names:
    is_imp = (n == "Фиолетовый")
    all_agents.append(Agent(n, is_impostor=is_imp))


running = True
thinking_cooldown = 0.05
visual_delay = 0.2
DEFAULT_THINK_COOLDOWN = 60

font_small = pygame.font.SysFont(None, 22)
REPORT_DISPLAY_TICKS = 600

while running:
    current_tick += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(map_image, (0, 0))

    for r in hitboxes:
        pygame.draw.rect(screen, HITBOX_COLOR, pygame.Rect(*r), 1)
    for room in rooms:
        rect = pygame.Rect(*room["rect"])
        pygame.draw.rect(screen, ROOM_COLOR, rect, 2)
        text = font_small.render(room["name"], True, ROOM_COLOR)
        screen.blit(text, (rect.x, rect.y))


    for ag in all_agents:
        if ag.think_cooldown > 0:
            ag.think_cooldown -= 1
        if getattr(ag, "patrol_cooldown", 0) > 0:
            ag.patrol_cooldown -= 1
            if ag.patrol_cooldown < 0:
                ag.patrol_cooldown = 0

  
    for ag in all_agents:
        if not ag.alive:
            continue
        if ag.current_task_index < len(ag.tasks) and not ag.current_path:
            next_task = ag.tasks[ag.current_task_index]
            try:
                next_room = next(r for r in rooms if r["name"].lower() == next_task['room'])
            except StopIteration:
                next_room = random.choice(rooms)
            target_pos = [
                next_room["rect"][0] + next_room["rect"][2] // 2,
                next_room["rect"][1] + next_room["rect"][3] // 2
            ]
            ag.current_path[:] = astar(ag.pos, target_pos)


    any_need_think = False
    for agent in all_agents:
        if not agent.alive:
            continue
        visibility_changed = agent.update_visibility(all_agents, corpses)
        needs = False
        if agent.think_cooldown == 0:
            needs = visibility_changed or agent.waiting_for_ai or (not agent.current_path and agent.current_task_index < len(agent.tasks))
        if is_in_crowd(agent, all_agents, crowd_size=3):
            needs = False
        agent.needs_think = needs
        if needs:
            any_need_think = True

    if any_need_think:
        thinkers = [a.color for a in all_agents if a.needs_think and a.alive]
        print(f"[Синхронная фаза] Думают: {thinkers}")
        decisions = {}
        for agent in all_agents:
            if not agent.alive or not agent.needs_think:
                continue
            nearby = [a for a in all_agents if a != agent and a.alive and a.color in agent.sees_agents]
            dec = ask_next_action(agent, nearby)
            decisions[agent.color] = dec
            agent.think_cooldown = DEFAULT_THINK_COOLDOWN
            time.sleep(thinking_cooldown)
            agent.needs_think = False

    
        for color, dec in decisions.items():
            actor = next((a for a in all_agents if a.color == color), None)
            if not actor or not actor.alive:
                continue
            if isinstance(dec, str) and dec.lower().startswith("kill_") and actor.is_impostor and actor.kill_cooldown == 0:
                target_name = dec.split("kill_")[1].strip().capitalize()
                target = next((a for a in all_agents if a.color.lower() == target_name.lower()), None)
                if target and target.alive and target.color != actor.color:
                    same_room = get_room_by_pos(actor.pos) == get_room_by_pos(target.pos)
                    near = ((actor.pos[0] - target.pos[0])**2 + (actor.pos[1] - target.pos[1])**2)**0.5 < 30
                    if same_room or near:
                        seen_by_others = any((target.color in a.sees_agents) for a in all_agents if a.color not in (actor.color, target.color) and a.alive)
                        if not seen_by_others:
                            perform_kill(actor, target)
                            break

 
        for color, dec in decisions.items():
            actor = next((a for a in all_agents if a.color == color), None)
            if not actor or not actor.alive:
                continue
            if isinstance(dec, str) and dec.lower().startswith("kill_"):
                actor.waiting_for_ai = False
                actor.think_cooldown = DEFAULT_THINK_COOLDOWN
                continue
            chosen_room = None
            if isinstance(dec, str):
                chosen_room = dec.lower()
            if not chosen_room:
                chosen_room = random.choice([r["name"].lower() for r in rooms])
            try:
                target_room = next(r for r in rooms if r["name"].lower() == chosen_room)
            except StopIteration:
                target_room = random.choice(rooms)
            target_pos = [target_room["rect"][0] + target_room["rect"][2] // 2,
                          target_room["rect"][1] + target_room["rect"][3] // 2]
            actor.current_path[:] = astar(actor.pos, target_pos)
            actor.waiting_for_ai = False
            actor.next_target_room_name = chosen_room
            actor.think_cooldown = DEFAULT_THINK_COOLDOWN
            print(f"[ACTION APPLY] {actor.color} идёт в {chosen_room}")

    #
    for agent in all_agents:
        if not agent.alive:
            continue

        if check_for_corpses(agent, corpses, all_agents):
            think_about_meeting(agent, corpses, all_agents)

        if agent.kill_cooldown > 0:
            agent.kill_cooldown -= 1

        if not agent.stopped_for_think:
            move_agent(agent)

        
        current_room = get_room_by_pos(agent.pos)
        if current_room == agent.next_target_room_name:
            agent.time_in_room += 1
        else:
            agent.time_in_room = 0
            agent.next_target_room_name = current_room

     
        if agent.current_task_index < len(agent.tasks):
            task = agent.tasks[agent.current_task_index]
            if get_room_by_pos(agent.pos) == task['room'] and not task['done']:
                task['done'] = True
                print(f"[TASK] {agent.color} выполнил задание в {task['room']}")
                agent.current_task_index += 1
                if agent.current_task_index < len(agent.tasks):
                    next_task = agent.tasks[agent.current_task_index]
                    next_room = next(r for r in rooms if r["name"].lower() == next_task['room'])
                    target_pos = [
                        next_room["rect"][0] + next_room["rect"][2] // 2,
                        next_room["rect"][1] + next_room["rect"][3] // 2
                    ]
                    agent.current_path[:] = astar(agent.pos, target_pos)

     
        if agent.current_task_index >= len(agent.tasks) and not agent.current_path and getattr(agent, "patrol_cooldown", 0) == 0:
            choose_new_room_ai(agent)

    #
    for agent in all_agents:
        if agent.alive:
            sprite = AGENT_SPRITES.get(agent.color)
            if sprite:
                screen.blit(sprite, (int(agent.pos[0]) - SPRITE_SIZE//2, int(agent.pos[1]) - SPRITE_SIZE//2))
            else:
                pygame.draw.circle(screen, AGENT_COLORS[agent.color], (int(agent.pos[0]), int(agent.pos[1])), agent.radius)
        else:
            pygame.draw.circle(screen, (100, 100, 100), (int(agent.pos[0]), int(agent.pos[1])), agent.radius)

    for corpse in corpses:
        pygame.draw.circle(screen, (50, 50, 50), (int(corpse['pos'][0]), int(corpse['pos'][1])), 6)

    if last_report:
        if current_tick - last_report["time"] < REPORT_DISPLAY_TICKS:
            text_str = f"REPORT: {last_report['reporter']} reported {last_report['victim']}"
            surf = font_small.render(text_str, True, (255, 255, 0))
            screen.blit(surf, (10, 10))
        else:
            last_report = None


    winner = check_win_condition()
    if winner:
        print("[GAME END] Winner:", winner)
        show_end_overlay(winner)
        end_time = time.time()
        while time.time() - end_time < 6:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            clock.tick(30)
        running = False
        continue

    pygame.display.flip()
    clock.tick(60)
    time.sleep(visual_delay)


print("Игра завершена. Логи мыслей Фиолетового сохранены в", PURPLE_LOG_FILENAME)
pygame.quit()
sys.exit()
