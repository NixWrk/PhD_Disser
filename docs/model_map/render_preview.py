"""Render this text-only Canvas to PNG; the .canvas remains the editable source.

Requires Pillow. Uses Windows Segoe UI, or a caller-supplied --font directory
containing segoeui.ttf, segoeuib.ttf and seguisym.ttf. Does not alter the Canvas.
"""

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts'))
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    canvas = json.loads((folder / 'Математическая_модель_эскиз.canvas').read_text(encoding='utf-8'))
    nodes = {node['id']: node for node in canvas['nodes']}
    margin = 45
    left = min(n['x'] for n in nodes.values()) - margin
    top = min(n['y'] for n in nodes.values()) - margin
    width = max(n['x'] + n['width'] for n in nodes.values()) - left + margin
    height = max(n['y'] + n['height'] for n in nodes.values()) - top + margin
    im = Image.new('RGB', (width, height), '#f5f7fb')
    draw = ImageDraw.Draw(im)
    fonts = {
        'body': ImageFont.truetype(str(args.font / 'segoeui.ttf'), 22),
        'status': ImageFont.truetype(str(args.font / 'segoeuib.ttf'), 18),
        'title': ImageFont.truetype(str(args.font / 'segoeuib.ttf'), 27),
        'page': ImageFont.truetype(str(args.font / 'segoeuib.ttf'), 38),
        'edge': ImageFont.truetype(str(args.font / 'segoeui.ttf'), 18),
        'group': ImageFont.truetype(str(args.font / 'segoeuib.ttf'), 17),
        'symbol': ImageFont.truetype(str(args.font / 'seguisym.ttf'), 22),
    }
    colors = {
        '1': ('#fff0ed', '#be4034'),
        '2': ('#fff7e5', '#a86c09'),
        '3': ('#fffbe1', '#8b7411'),
        '4': ('#ecf8f0', '#28744a'),
        '5': ('#edf5ff', '#28659d'),
        '6': ('#f3effd', '#7952a2'),
    }

    def xy(x, y):
        return (x - left, y - top)

    def wrap(text, font, limit):
        lines = []
        for para in text.splitlines():
            if not para:
                lines.append('')
                continue
            line = ''
            for word in para.split():
                trial = (line + ' ' + word).strip()
                if line and draw.textlength(trial, font=font) > limit:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
        return lines

    def lines(text, font, x, y, limit, color, step):
        for line in wrap(text, font, limit):
            if line:
                xx = x
                for part in re.split('(∇)', line):
                    chosen = fonts['symbol'] if part == '∇' else font
                    draw.text(xy(xx, y), part, font=chosen, fill=color)
                    xx += draw.textlength(part, font=chosen)
                y += step
            else:
                y += step * 0.35
        return y

    def port(node_id, side):
        n = nodes[node_id]
        x, y, w, h = n['x'], n['y'], n['width'], n['height']
        return {'top': (x + w / 2, y), 'right': (x + w, y + h / 2),
                'bottom': (x + w / 2, y + h), 'left': (x, y + h / 2)}[side]

    for n in nodes.values():
        if n['type'] != 'group':
            continue
        x, y, w, h = n['x'], n['y'], n['width'], n['height']
        draw.rounded_rectangle([xy(x, y), xy(x+w, y+h)], radius=18,
                               fill='#edf0f5', outline='#d8deea', width=2)
        draw.text(xy(x+12, y+8), n['label'], font=fonts['group'], fill='#536174')

    # The standard has endpoint sides, not explicit bend points. Curves below
    # are a preview; Obsidian chooses its own curves from the same endpoints.
    label_positions = []
    vectors = {'left': (-1, 0), 'right': (1, 0), 'top': (0, -1), 'bottom': (0, 1)}
    for edge in canvas['edges']:
        start, end = port(edge['fromNode'], edge['fromSide']), port(edge['toNode'], edge['toSide'])
        dist = min(180, max(45, math.dist(start, end) * .35))
        if edge['fromSide'] == 'right' and edge['toSide'] == 'left' and end[0] > start[0]:
            dist = min(dist, (end[0]-start[0])/2)
        elif edge['fromSide'] == 'bottom' and edge['toSide'] == 'top' and end[1] > start[1]:
            dist = min(dist, (end[1]-start[1])/2)
        elif edge['fromSide'] == 'top' and edge['toSide'] == 'bottom' and start[1] > end[1]:
            dist = min(dist, (start[1]-end[1])/2)
        sv, ev = vectors[edge['fromSide']], vectors[edge['toSide']]
        c1 = (start[0]+dist*sv[0], start[1]+dist*sv[1])
        c2 = (end[0]+dist*ev[0], end[1]+dist*ev[1])
        points = []
        for i in range(101):
            t = i/100
            points.append(((1-t)**3*start[0]+3*(1-t)**2*t*c1[0]+3*(1-t)*t*t*c2[0]+t**3*end[0],
                           (1-t)**3*start[1]+3*(1-t)**2*t*c1[1]+3*(1-t)*t*t*c2[1]+t**3*end[1]))
        color = colors[edge['color']][1]
        draw.line([xy(*p) for p in points], fill=color, width=3)
        theta = math.atan2(end[1]-points[-4][1], end[0]-points[-4][0])
        arrow = [end, (end[0]-14*math.cos(theta-.45), end[1]-14*math.sin(theta-.45)),
                 (end[0]-14*math.cos(theta+.45), end[1]-14*math.sin(theta+.45))]
        draw.polygon([xy(*p) for p in arrow], fill=color)
        label_positions.append((edge, points[50], color))

    for n in nodes.values():
        if n['type'] != 'text':
            continue
        x, y, w, h = n['x'], n['y'], n['width'], n['height']
        if n['id'] in ('title', 'legend'):
            text = n['text'].replace('# ', '').replace('**', '')
            if n['id'] == 'title':
                heading, body = text.split('\n\n', 1)
                yy = lines(heading, fonts['page'], x+5, y+8, w-10, '#172a45', 48)
                lines(body, fonts['body'], x+5, yy+8, w-10, '#40516a', 29)
            else:
                lines(text, fonts['body'], x+5, y+12, w-10, '#40516a', 29)
            continue
        fill, stroke = colors[n['color']]
        draw.rounded_rectangle([xy(x, y), xy(x+w, y+h)], radius=15, fill=fill, outline=stroke, width=2)
        title, status, rest = n['text'].split('\n\n', 2)
        body, source = rest.rsplit('\n\n', 1)
        title = title.removeprefix('## ')
        status = status.replace('**', '')
        yy = lines(title, fonts['title'], x+22, y+17, w-44, '#172a45', 34)
        yy = lines(status, fonts['status'], x+22, yy+10, w-44, stroke, 24)
        yy = lines(body, fonts['body'], x+22, yy+14, w-44, '#26384e', 29)
        if yy > y+h-42:
            raise ValueError(f'Text overflow in {n["id"]}: {yy-y:.1f}px')
        source = re.sub(r'\[\[[^|]+\|([^]]+)\]\]', r'\1', source)
        draw.text(xy(x+22, y+h-31), source+' · подробнее в заметке', font=fonts['edge'], fill=stroke)

    # Keep labels short and wrapped inside gaps between cards.
    for edge, p, color in label_positions:
        a, b = nodes[edge['fromNode']], nodes[edge['toNode']]
        vertical = edge['fromSide'] in ('top', 'bottom') and edge['toSide'] in ('top', 'bottom')
        same_row = a['y'] == b['y']
        max_width = 160 if vertical else 110
        label_lines = wrap(edge['label'], fonts['edge'], max_width)
        lw = max(draw.textlength(s, font=fonts['edge']) for s in label_lines) + 14
        lh = len(label_lines)*23+8
        # Horizontal labels sit above the edge, clear of card text.
        lx = p[0]-lw/2
        ly = p[1]-lh-10 if same_row else p[1]-lh/2
        # A narrow horizontal gap cannot contain a long label. Place it just
        # above the two cards, as a caption to the adjacent connection.
        if same_row and b['x']-a['x']-a['width'] < lw:
            max_width = 250
            label_lines = wrap(edge['label'], fonts['edge'], max_width)
            lw = max(draw.textlength(s, font=fonts['edge']) for s in label_lines)+14
            lh = len(label_lines)*23+8
            lx, ly = p[0]-lw/2, a['y']-lh-5
        draw.rounded_rectangle([xy(lx, ly), xy(lx+lw, ly+lh)], radius=5, fill='#ffffff', outline='#d8deea')
        for i, line in enumerate(label_lines):
            draw.text(xy(lx+7, ly+3+i*23), line, font=fonts['edge'], fill=color)
    output = folder / 'Математическая_модель_обзор.png'
    im.save(output, optimize=True)
    print(f'{output.name}: {width}x{height}; no text overflow')


if __name__ == '__main__':
    main()
