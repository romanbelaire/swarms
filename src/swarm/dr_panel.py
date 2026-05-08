"""
Pygame side-panel for DR mixture gates (u, v, joint w) during play visualization.
"""

import pygame


def _draw_bar_row(surface, font, rect, values, labels, title: str):
    pygame.draw.rect(surface, (240, 240, 240), rect)
    pygame.draw.rect(surface, (100, 100, 100), rect, 1)
    y = rect.y + 2
    surface.blit(font.render(title, True, (0, 0, 0)), (rect.x + 4, y))
    y += font.get_height() + 2
    n = len(values)
    bar_w = (rect.w - 8) // n
    for i in range(n):
        bx = rect.x + 4 + i * bar_w
        bw = bar_w - 4
        bh = 10
        pygame.draw.rect(surface, (200, 200, 200), (bx, y, bw, bh))
        fill_w = int(bw * float(values[i]))
        pygame.draw.rect(surface, (60, 120, 200), (bx, y, fill_w, bh))
        surface.blit(font.render(labels[i], True, (40, 40, 40)), (bx, y + bh + 1))


def _draw_w_heatmap(surface, font, rect, w, scenario_delta):
    pygame.draw.rect(surface, (250, 250, 250), rect)
    pygame.draw.rect(surface, (100, 100, 100), rect, 1)
    cell = min((rect.w - 8) // 3, (rect.h - 24) // 3)
    cell = max(cell, 8)
    ox = rect.x + 4
    oy = rect.y + 4
    surface.blit(font.render("w (role x others)", True, (0, 0, 0)), (ox, oy))
    oy += font.get_height() + 2
    flat_arg = int(w.argmax())
    ar, ac = flat_arg // 3, flat_arg % 3
    for r in range(3):
        for c in range(3):
            val = float(w[r, c])
            gray = int(255 * val)
            gray = max(0, min(255, gray))
            cx = ox + c * cell
            cy = oy + r * cell
            cr = pygame.Rect(cx, cy, cell - 1, cell - 1)
            pygame.draw.rect(surface, (gray, gray, gray), cr)
            if r == ar and c == ac:
                pygame.draw.rect(surface, (220, 50, 50), cr, 2)
            else:
                pygame.draw.rect(surface, (120, 120, 120), cr, 1)
    d_at_arg = float(scenario_delta[ar, ac])
    oy += 3 * cell + 4
    surface.blit(font.render(f"D at argmax(w): {d_at_arg:.3f}", True, (0, 0, 0)), (ox, oy))


def render_dr_panel(
    canvas: pygame.Surface,
    panel_rect: pygame.Rect,
    per_agent_rows: list[dict],
    env_reward: float,
    team_utility: float,
    step_idx: int,
    my_mean_by_agent: dict[str, float],
):
    """
    per_agent_rows: each dict has keys agent_id, u (3,), v (3,), w (3,3), scenario_delta (3,3), mixed_reward.
    """
    font = pygame.font.SysFont(None, 16)
    small = pygame.font.SysFont(None, 14)
    pygame.draw.rect(canvas, (230, 230, 235), panel_rect)
    pygame.draw.line(canvas, (80, 80, 80), (panel_rect.left, panel_rect.top), (panel_rect.left, panel_rect.bottom), 2)

    header_h = 28
    hr = pygame.Rect(panel_rect.x + 4, panel_rect.y + 4, panel_rect.w - 8, header_h)
    pygame.draw.rect(canvas, (210, 215, 225), hr)
    t = f"step {step_idx}  env_R={env_reward:.2f}  team_util={team_utility:.2f}"
    canvas.blit(font.render(t, True, (0, 0, 0)), (hr.x + 4, hr.y + 6))

    n_agents = len(per_agent_rows)
    if n_agents == 0:
        return

    body_top = panel_rect.y + header_h + 12
    body_h = panel_rect.h - header_h - 16
    slot_h = body_h // n_agents

    u_labels = ("solv", "neut", "caus")
    v_labels = ("AllC", "AllP", "Same")

    for idx, row in enumerate(per_agent_rows):
        agent_id = row["agent_id"]
        u = row["u"]
        v = row["v"]
        w = row["w"]
        scenario_delta = row["scenario_delta"]
        r_mix = row["mixed_reward"]
        mm = my_mean_by_agent[agent_id]

        slot_y = body_top + idx * slot_h
        slot = pygame.Rect(panel_rect.x + 4, slot_y, panel_rect.w - 8, slot_h - 4)
        pygame.draw.rect(canvas, (245, 245, 248), slot)
        pygame.draw.rect(canvas, (160, 160, 170), slot, 1)

        ty = slot.y + 4
        title = f"{agent_id}  R_mix={r_mix:.3f}  my_mean={mm:.3f}"
        canvas.blit(font.render(title, True, (20, 20, 40)), (slot.x + 4, ty))
        ty += font.get_height() + 4

        bar_row_h = 38
        u_rect = pygame.Rect(slot.x + 4, ty, slot.w - 8, bar_row_h)
        _draw_bar_row(canvas, small, u_rect, u, u_labels, "u (role)")
        ty += bar_row_h + 4

        v_rect = pygame.Rect(slot.x + 4, ty, slot.w - 8, bar_row_h)
        _draw_bar_row(canvas, small, v_rect, v, v_labels, "v (others)")
        ty += bar_row_h + 4

        hm_h = 3 * 14 + 40
        hm_rect = pygame.Rect(slot.x + 4, ty, slot.w - 8, hm_h)
        _draw_w_heatmap(canvas, small, hm_rect, w, scenario_delta)
