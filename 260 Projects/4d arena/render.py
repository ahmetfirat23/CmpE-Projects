import time
import sys
import subprocess
import tkinter as tk
import re


class Grid:
    def __init__(self, num_squares, square_size, padding, offset, fps=10):
        self.row_i = 0
        self.col_i = 0
        self.max_col = 3

        self.fps = fps
        self.num_squares = num_squares
        self.square_size = square_size
        self.padding = padding
        self.offset = offset
        self.height = num_squares[0]*square_size+(num_squares[0]-1)*padding + offset*2
        self.width = num_squares[1]*square_size+(num_squares[1]-1)*padding + offset*2

        self.root = tk.Tk()
        self.root.title("4D Arena")
        self.root.geometry("%dx%d" % (self.max_col*self.width, 2*self.height))
        outer_frame = tk.Frame(self.root)
        outer_frame.pack(fill="both", expand=True)
        self._out_canvas = tk.Canvas(outer_frame, bg="#111111")
        self._out_canvas.pack(side="left", fill="both", expand=True)
        self._out_canvas.bind_all("<MouseWheel>", lambda e: self._out_canvas.yview_scroll(int(-1*(e.delta)), "units"))
        self._out_canvas.bind_all("<Button-4>", lambda e: self._out_canvas.yview_scroll(-1, "units"))
        self._out_canvas.bind_all("<Button-5>", lambda e: self._out_canvas.yview_scroll(1, "units"))
        self._out_canvas.bind_all("<Shift-MouseWheel>", lambda e: self._out_canvas.xview_scroll(int(-1*(e.delta)), "units"))
        self._out_canvas.bind_all("<Shift-Button-4>", lambda e: self._out_canvas.xview_scroll(-1, "units"))
        self._out_canvas.bind_all("<Shift-Button-5>", lambda e: self._out_canvas.xview_scroll(1, "units"))

        self.inner_frame = tk.Frame(self._out_canvas)
        self._out_canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.frame_title = {}
        self.canvas = {}
        self.rectangles = {}
        self.previous_squares = {}
        subsample_rate = int(32 / square_size)
        self.sprites = {"warrior": tk.PhotoImage(file="warrior.png").subsample(subsample_rate, subsample_rate),
                        "rogue": tk.PhotoImage(file="rogue.png").subsample(subsample_rate, subsample_rate),
                        "wizard": tk.PhotoImage(file="wizard.png").subsample(subsample_rate, subsample_rate),
                        "grass": tk.PhotoImage(file="grass.png").subsample(subsample_rate, subsample_rate)}

    def initialize_new_window(self, universe_id):
        frame = tk.Frame(self.inner_frame)
        frame.grid(row=self.row_i, column=self.col_i)
        self.col_i += 1
        if self.col_i == self.max_col:
            self.row_i += 1
            self.col_i = 0
        canvas = tk.Canvas(frame, height=self.height-5, width=self.width-5, bg="#8B4513")
        canvas.pack()
        self.canvas[universe_id] = canvas
        self.rectangles[universe_id] = []
        self.previous_squares[universe_id] = {}

    def draw(self, universe_id):
        canvas = self.canvas[universe_id]
        for i in range(self.num_squares[0]):
            row_rect = []
            for j in range(self.num_squares[1]):
                rect = canvas.create_rectangle(
                    j*(self.square_size+self.padding)+self.offset,
                    i*(self.square_size+self.padding)+self.offset,
                    (j+1)*self.square_size+j*self.padding+self.offset,
                    (i+1)*self.square_size+i*self.padding+self.offset,
                    fill="#000000")
                sprite = canvas.create_image(
                    j*(self.square_size+self.padding)+self.offset+self.square_size/2,
                    i*(self.square_size+self.padding)+self.offset+self.square_size/2,
                    image=self.sprites["grass"])
                row_rect.append((rect, sprite))
            self.rectangles[universe_id].append(row_rect)
        self.frame_title[universe_id] = canvas.create_text(
            self.width/2,
            11,
            text=f"Universe: {universe_id}",
            font="%d" % (self.square_size//2),
            fill="#FFFFFF")

    def update(self, agents, universe_id, t, turn):
        if universe_id not in self.canvas:
            self.initialize_new_window(universe_id)
            self.draw(universe_id)
        canvas = self.canvas[universe_id]
        previous_squares = self.previous_squares[universe_id]
        for key in previous_squares:
            canvas.delete(previous_squares[key][0])
            canvas.delete(previous_squares[key][1])
            canvas.delete(previous_squares[key][2])
            canvas.delete(previous_squares[key][3])
            canvas.delete(previous_squares[key][4])
            canvas.delete(previous_squares[key][5])
        for agent_id, (agent_class, name, health, mana, x, y) in agents.items():
            y = self.num_squares[0] - y - 1
            health_percent = int((health/100)*self.square_size)
            mana_percent = int((mana/100)*self.square_size)
            previous_squares[agent_id] = (canvas.create_image(
                x*(self.square_size+self.padding)+self.offset+self.square_size/2,
                y*(self.square_size+self.padding)+self.offset+self.square_size/2,
                image=self.sprites[agent_class]),
                                          canvas.create_rectangle(
                x*(self.square_size+self.padding)+self.offset,
                y*(self.square_size+self.padding)+self.offset-6,
                (x+1)*self.square_size+x*self.padding+self.offset,
                y*(self.square_size+self.padding)+self.offset-3,
                fill="#FF0000",
                width=1),
                                          canvas.create_rectangle(
                x*(self.square_size+self.padding)+self.offset,
                y*(self.square_size+self.padding)+self.offset-6,
                x*(self.square_size+self.padding)+self.offset+health_percent,
                y*(self.square_size+self.padding)+self.offset-3,
                fill="#00FF00",
                width=1),
                                          canvas.create_rectangle(
                x*(self.square_size+self.padding)+self.offset,
                y*(self.square_size+self.padding)+self.offset-3,
                (x+1)*self.square_size+x*self.padding+self.offset,
                y*(self.square_size+self.padding)+self.offset,
                fill="#FF0000",
                width=1),
                                          canvas.create_rectangle(
                x*(self.square_size+self.padding)+self.offset,
                y*(self.square_size+self.padding)+self.offset-3,
                x*(self.square_size+self.padding)+self.offset+mana_percent,
                y*(self.square_size+self.padding)+self.offset,
                fill="#0000FF",
                width=1),
                                          canvas.create_text(
                x*(self.square_size+self.padding)+self.offset,
                (y+1)*(self.square_size+self.padding)+self.offset-5,
                text=name,
                anchor="w",
                font="sans-serif %d" % (self.square_size//4),
                fill="#FFFFFF")
            )

        self.root.update()
        num_total_agents = 0
        for key in self.previous_squares:
            num_total_agents += len(self.previous_squares[key])
        sleep_dur = (1/(self.fps*num_total_agents))
        time.sleep(sleep_dur)
        self.canvas[universe_id].itemconfig(self.frame_title[universe_id], text=f"Time: {t}, Universe: {universe_id}, Turn: {turn}")


def get_result(kb, query):
    out = subprocess.run(["swipl", "-g", f"consult('main.pro'), consult('{kb}').", "-g", query, "-g", "halt"], capture_output=True)
    return out.stdout.decode("utf-8"), out.stderr.decode("utf-8")


def generate_data(kb, timesteps):
    query = "main_loop(%d), \
            findall(StateId-Agents-Universe-Time-Turn, \
                    (state(StateId, Agents, _, _), \
                    history(StateId, Universe, Time, Turn)), \
                    Buffer), \
            print_array(Buffer)" % timesteps
    out, _ = get_result(kb, query)
    lines = out.strip("[]\n").split()
    render_data = []
    for line in lines:
        state_id, agent_dict, universe_id, t, turn = line.split("-")
        state_id = int(state_id)
        universe_id = int(universe_id)
        t = int(t)
        turn = int(turn)
        agent_dict = re.findall(r"(\d+):agent{agility:(\d+),armor:(\d+),class:(\w+),health:(\d+),mana:(\d+),name:(\w+),x:(\d+),y:(\d+)}", agent_dict)
        agents = {}
        for agent in agent_dict:
            agent_id, agility, armor, agent_class, health, mana, name, x, y = agent
            agent_id = int(agent_id)
            agility = int(agility)
            armor = int(armor)
            health = int(health)
            mana = int(mana)
            x = int(x)
            y = int(y)
            agents[agent_id] = (agent_class, name, health, mana, x, y)
        render_data.append((state_id, agents, universe_id, t, turn))
    return render_data


if __name__ == "__main__":
    render_data = generate_data(sys.argv[1], int(sys.argv[2]))
    print("Generated")
    grid = Grid((10, 10), 32, 2, 20, fps=60)
    for state_id, agents, universe_id, t, turn in render_data:
        print(f"Uni={universe_id}, Sid={state_id}, t={t}, turn={turn}, agents={agents}")
        grid.update(agents, universe_id, t, turn)
    grid.root.mainloop()
