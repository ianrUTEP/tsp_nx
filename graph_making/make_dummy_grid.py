import json

def generate_nodes(
    rows=5,
    cols=4,
    x_start=20.0,
    x_inner_offset=0.8,
    x_far_offset=19.2,
    x_end=40.0,
    y_start=20.0,
    y_step=10.0
):
    nodes = {}
    node_id = 1

    for r in range(1, rows + 1):
        y = y_start + (r - 1) * y_step

        x_positions = [
            x_start,
            x_start + x_inner_offset,
            x_start + x_far_offset,
            x_end
        ]

        for c in range(1, cols + 1):
            nodes[str(node_id)] = {
                "pos": [x_positions[c - 1], y],
                "streamline": [r, c],
                "CompV": [1, 0, 0] if r == 1 else [0.6, 0.4, 0],
                "pStressV": [0, 0],
                "partMember": 0 if (c == 1 or c == cols) else 1
            }
            node_id += 1

    return [{"nodes": nodes}]



data = generate_nodes()
print(json.dumps(data, indent=2))