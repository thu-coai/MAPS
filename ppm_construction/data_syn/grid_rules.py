import os
import json 
import numpy as np
import random
np.random.seed(42)
random.seed(42)
import readchar

# NOTE: Components Types
(
    TYPE_SHORT,
    TYPE_VOLTAGE_SOURCE,
    TYPE_CURRENT_SOURCE,
    TYPE_RESISTOR,
    TYPE_CAPACITOR,
    TYPE_INDUCTOR,

    TYPE_OPEN, # Open Circuit
    TYPE_VCCS, # Voltage-Controlled Current Source --> G in SPICE
    TYPE_VCVS, # Voltage-Controlled Voltage Source --> E in SPICE
    TYPE_CCCS, # Current-Controlled Current Source --> F in SPICE
    TYPE_CCVS, # Current-Controlled Voltage Source --> H in SPICE
) = tuple( range(11) )
NUM_NORMAL=6

# NOTE: Type of Measurements
(
    MEAS_TYPE_NONE,
    MEAS_TYPE_VOLTAGE,
    MEAS_TYPE_CURRENT,
) = tuple( range(3) )

# NOTE: TYPE of Units
(
    UNIT_MODE_1,
    UNIT_MODE_k,
    UNIT_MODE_m,
    UNIT_MODE_u,
    UNIT_MODE_n,
    UNIT_MODE_p,
) = tuple( range(6) )

# NOTE: LATEX formatting
vlt7_latex_template = r"""\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{circuitikz}
\begin{document}
\begin{center}
\begin{circuitikz}[line width=1pt]
\ctikzset{tripoles/en amp/input height=0.5};
\ctikzset{inductors/scale=1.2, inductor=american}
<main>
\end{circuitikz}
\end{center}
\end{document}"""
v8_latex_template = r"""\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{circuitikz}
\tikzset{every node/.style={font=<font>}}
\tikzset{every draw/.style={font=<font>}}
\begin{document}
\begin{center}
\begin{circuitikz}[line width=1pt]
\ctikzset{tripoles/en amp/input height=0.5};
\ctikzset{inductors/scale=1.2, inductor=american}
<main>
\end{circuitikz}
\end{center}
\end{document}"""
v8_latex_template = r"""\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{circuitikz}
\tikzset{every node/.style={font=<font>}}
\tikzset{every draw/.style={font=<font>}}
\begin{document}
\begin{center}
\begin{circuitikz}[line width=1pt]
\ctikzset{tripoles/en amp/input height=0.5};
\ctikzset{inductors/scale=1.2, inductor=american}
<main>
\end{circuitikz}
\end{center}
\end{document}"""

LATEX_TEMPLATES = {
    "v<=7": vlt7_latex_template,
    "v8": v8_latex_template,
    "v9": v8_latex_template,
    "v10": v8_latex_template,
    "v11": v8_latex_template,
}

unit_scales = ["", "k", "m", "\\mu", "n", "p"]

LABEL_TYPE_NUMBER, LABEL_TYPE_STRING = tuple(range(2)) # label is numerical format or string format
components_latex_info = [("short", "", ""), ("V","U","V"), ("I","I","A"), ("generic","R","\Omega"), ("C","C","F"), ("L","L","H"),
                         ("open", "", ""), ("cisource", "", ""), ("cvsource", "", ""), ("cisource", "", ""), ("cvsource", "", "") ] # type, label, unit

CUR_MODE_1, CUR_MODE_2, CUR_MODE_3, CUR_MODE_4, CUR_MODE_5, CUR_MODE_6 = tuple(range(6))
flow_direction = ["^>", ">_", "^>", "_>"]

def get_latex_line_draw(x1, y1, x2, y2,
                        type_number, 
                        label_subscript,
                        value, 
                        value_unit,
                        use_value_annotation,   # True: annotate value in the figure / False: annotate label in the figure
                        style="chinese",
                        measure_type=MEAS_TYPE_NONE,
                        measure_label="",
                        measure_direction=0,
                        control_label="",
                        label_subscript_type=LABEL_TYPE_NUMBER,
                        direction=0,
                        note='v5'
                    ) -> str:
    
    if direction == 1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    meas_comp_same_direction = (direction == measure_direction)
    
    if style == "chinese":
        print(f"drawing between ({x1:.1f},{y1:.1f}) and ({x2:.1f},{y2:.1f})\n")
        print(f"type_num: {type_number}, label_num: {label_subscript}, value: {value}, use_value_annotation: {use_value_annotation}, label_type_number: {label_subscript_type}, direction: {direction}")
        print(f"measure_type: {measure_type}, measure_label: {measure_label}, measure_direction: {measure_direction}")
        type_number = int(type_number)
        
        comp_circuitikz_type = components_latex_info[type_number][0]
        comp_label_main = components_latex_info[type_number][1]
        comp_standard_unit = components_latex_info[type_number][2]

        # NOTE: Get the label of the component
        labl = ""
        if control_label == -1: control_label = ""
        control_label = f"_{control_label}" if control_label != "" else ""
        if use_value_annotation:    # numerical-type circuit
            if type_number < NUM_NORMAL:
                real_value = value
                unit_mode = value_unit
                unit_scale = unit_scales[unit_mode]
                if int(note[1:]) <= 9:
                    raise NotImplementedError
                elif int(note[1:]) > 9:
                    labl = f"{int(real_value)} \\mathrm{{ {unit_scale}{comp_standard_unit} }}"
            else:
                if type_number == TYPE_VCCS or type_number == TYPE_VCVS:
                    labl = f"{value} U_{{ {control_label} }}"
                elif type_number == TYPE_CCCS or type_number == TYPE_CCVS:
                    labl = f"{value} I_{{ {control_label} }}"

        else:       # label-type circuit
            if type_number < NUM_NORMAL:
                if label_subscript_type == LABEL_TYPE_NUMBER:
                    if type_number == TYPE_RESISTOR:
                        labl = f"{comp_label_main}_{{ {int(label_subscript)} }}" # e.g. R_{1}
                    elif type_number == TYPE_VOLTAGE_SOURCE or type_number == TYPE_CURRENT_SOURCE:
                        labl = f"{comp_label_main}_{{ S{int(label_subscript)} }}" # e.g. U_{S1}

                elif label_subscript_type == LABEL_TYPE_STRING:
                    labl = f"{comp_label_main}_{{ {label_subscript} }}" # e.g. R_{load}
            
            else:
                if type_number == TYPE_VCCS or type_number == TYPE_VCVS:
                    labl = f"\\beta_{{ {label_subscript} }} U_{{ {control_label} }}"
                elif type_number == TYPE_CCCS or type_number == TYPE_CCVS:
                    labl = f"\\alpha_{{ {label_subscript} }} I_{{ {control_label} }}"

        print(f'labl: {labl}')

        # NOTE: get the label of measurement
        if measure_label == -1: measure_label = ""
        measure_label = f"_{{{str(measure_label)}}}" if measure_label != "" else ""
        if measure_type == MEAS_TYPE_CURRENT:
            measure_label = f"I{measure_label}"
        elif measure_type == MEAS_TYPE_VOLTAGE:
            measure_label = f"U{measure_label}"
        
# NOTE: Plot the components 
            
# NOTE: plot the shorcut
        if type_number == TYPE_SHORT:
            ret = f"\\draw ({x1:.1f},{y1:.1f}) to[short] ({x2:.1f},{y2:.1f});\n"
            
            if not meas_comp_same_direction:
                    x1, y1, x2, y2 = x2, y2, x1, y1
            
            if measure_type == MEAS_TYPE_CURRENT:
                flow_dir = flow_direction[np.random.choice(range(4))]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            print(f"ret: {ret}")
            return ret
        
# NOTE: plot the voltage source
        elif type_number == TYPE_VOLTAGE_SOURCE:
            if int(note[1:]) < 8:
                ret =  f"\\draw ({x1:.1f},{y1:.1f}) to[V] ({x2:.1f},{y2:.1f});\n\\ctikzset{{american}}\n\\draw ({x1:.1f},{y1:.1f}) to [short, v=${labl}$] ({x2:.1f},{y2:.1f});\n\\ctikzset{{european}}\n"
            else:
                ret =  f"\\draw ({x1:.1f},{y1:.1f}) to [short] ({x2:.1f},{y2:.1f});\n\\ctikzset{{american}};\n\\draw ({x1:.1f},{y1:.1f}) to[rmeter, t, v=${labl}$] ({x2:.1f},{y2:.1f});\n\\ctikzset{{european}};\n"

            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
            if measure_type == MEAS_TYPE_CURRENT:
                flow_dir = flow_direction[np.random.choice(range(4))]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[rmeter, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            
            return ret

# NOTE: plot the current source
        elif type_number == TYPE_CURRENT_SOURCE:
            if int(note[1:]) >= 8:
                cur_mode_choices = [CUR_MODE_1, CUR_MODE_2] * 10 + [CUR_MODE_3, CUR_MODE_4] * 0 + [CUR_MODE_5, CUR_MODE_6] * 1
                cur_mode = np.random.choice(cur_mode_choices)
                print(f"cur_mode: {cur_mode} when ploting current source")
            else:
                cur_mode == CUR_MODE_1
            
            ret = f"\\draw ({x1:.1f},{y1:.1f}) to[I] ({x2:.1f},{y2:.1f});\n"

            if cur_mode == CUR_MODE_1 or cur_mode == CUR_MODE_2:
                mid = np.array([(x1+x2)/2, (y1+y2)/2])
                vector = np.array([x2-x1, y2-y1])
                normal = np.array([-vector[1], vector[0]], dtype=np.float64)
                normal /= np.linalg.norm(normal)
                if cur_mode == CUR_MODE_1:
                    new_mid = mid + 0.6*normal
                    new_mid_node = mid + normal

                else:
                    new_mid = mid - 0.6*normal
                    new_mid_node = mid - normal

                norm_vector = vector / np.linalg.norm(vector)
                new_start = new_mid - 0.4*norm_vector
                new_end = new_mid + 0.4*norm_vector
                ret += f"\\draw[-latexslim] ({new_start[0]:.1f},{new_start[1]:.1f}) to ({new_end[0]:.1f},{new_end[1]:.1f});\n"
                ret += f"\\node at ({new_mid_node[0]:.1f}, {new_mid_node[1]:.1f}) {{${labl}$}};\n"
                
            elif cur_mode in [CUR_MODE_3, CUR_MODE_4, CUR_MODE_5, CUR_MODE_6]:
                flow_dir = flow_direction[cur_mode-2]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[rmeter, f{flow_dir}=${labl}$] ({x2:.1f},{y2:.1f});\n"
            
            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret += f"\\ctikzset{{american}}\n\\draw ({x1:.1f},{y1:.1f}) to[rmeter, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n\\ctikzset{{european}}\n"
                
            return ret

# NOTE: Plot resistance, capacitance & inductance
        elif type_number in [TYPE_RESISTOR, TYPE_CAPACITOR, TYPE_INDUCTOR]:
            ret = f"\\draw ({x1:.1f},{y1:.1f}) to[{comp_circuitikz_type}, l=${labl}$, ] ({x2:.1f},{y2:.1f});\n"

            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret +=  f"\\ctikzset{{american}}\n\\draw ({x1:.1f},{y1:.1f}) to[{comp_circuitikz_type}, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n\\ctikzset{{european}}\n"

            elif measure_type == MEAS_TYPE_CURRENT:
                if int(note[1:]) >= 8:
                    cur_mode_choices = [CUR_MODE_1, CUR_MODE_2] * 0 + [CUR_MODE_3, CUR_MODE_4] * 1 + [CUR_MODE_5, CUR_MODE_6] * 1
                    cur_mode = np.random.choice(cur_mode_choices)
                else:
                    cur_mode == CUR_MODE_5

                if cur_mode in [CUR_MODE_1, CUR_MODE_2]:
                    # ret = f"\\draw ({x1:.1f},{y1:.1f}) to[{comp_circuitikz_type}, l=${labl}$] ({x2:.1f},{y2:.1f});\n"
                    mid = np.array([(x1+x2)/2, (y1+y2)/2])
                    vector = np.array([x2-x1, y2-y1])
                    normal = np.array([-vector[1], vector[0]], dtype=np.float64)
                    normal /= np.linalg.norm(normal)
                    if cur_mode == CUR_MODE_1:
                        new_mid = mid + 0.4*normal
                        new_mid_node = mid + 0.8*normal

                    else:
                        new_mid = mid - 0.4*normal
                        new_mid_node = mid - 0.8*normal

                    norm_vector = vector / np.linalg.norm(vector)
                    new_start = new_mid - 0.4*norm_vector
                    new_end = new_mid + 0.4*norm_vector
                    ret += f"\\draw[-latexslim] ({new_start[0]:.1f},{new_start[1]:.1f}) to ({new_end[0]:.1f},{new_end[1]:.1f});\n"
                    ret += f"\\node at ({new_mid_node[0]:.1f},{new_mid_node[1]:.1f}) {{${labl}$}};\n"
                
                elif cur_mode in [CUR_MODE_3, CUR_MODE_4, CUR_MODE_5, CUR_MODE_6]:
                    flow_dir = flow_direction[cur_mode-2]
                    ret += f"\\draw ({x1:.1f},{y1:.1f}) to[{comp_circuitikz_type}, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n" 

            return ret

# NOTE: plot open circuit
        elif type_number == TYPE_OPEN:
            ret = ""

            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret += f"\\ctikzset{{american}};\n\\draw ({x1:.1f},{y1:.1f}) to[open, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n\\ctikzset{{european}};\n"
            return ret
        
# NOTE: plot controlled voltage source
        elif type_number in [TYPE_VCVS, TYPE_CCVS]:
            ret = f"\\ctikzset{{american}};\n\\draw ({x1:.1f},{y1:.1f}) to [short, v=${labl}$] ({x2:.1f},{y2:.1f});\n\\ctikzset{{european}};\n\\draw ({x1:.1f},{y1:.1f}) to[cvsource] ({x2:.1f},{y2:.1f});"

            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
            if measure_type == MEAS_TYPE_CURRENT:
                flow_dir = flow_direction[np.random.choice(range(4))]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[short, f{flow_dir}=${measure_label}$] ({x2:.1f},{y2:.1f});\n"
            
            return ret

# NOTE：plot controlled current source
        elif type_number in [TYPE_VCCS, TYPE_CCCS]:
            ret = f"\\draw ({x1:.1f},{y1:.1f}) to[cisource] ({x2:.1f},{y2:.1f});\n"

            cur_mode_choices = [CUR_MODE_1, CUR_MODE_2] * 10 + [CUR_MODE_3, CUR_MODE_4] * 0 + [CUR_MODE_5, CUR_MODE_6] * 1
            cur_mode = np.random.choice(cur_mode_choices)
            print(f"cur_mode: {cur_mode} when ploting current source")

            if cur_mode == CUR_MODE_1 or cur_mode == CUR_MODE_2:
                mid = np.array([(x1+x2)/2, (y1+y2)/2])
                vector = np.array([x2-x1, y2-y1])
                normal = np.array([-vector[1], vector[0]], dtype=np.float64)
                normal /= np.linalg.norm(normal)
                if cur_mode == CUR_MODE_1:
                    new_mid = mid + 0.6*normal
                    new_mid_node = mid + normal

                else:
                    new_mid = mid - 0.6*normal
                    new_mid_node = mid - normal

                norm_vector = vector / np.linalg.norm(vector)
                new_start = new_mid - 0.4*norm_vector
                new_end = new_mid + 0.4*norm_vector
                ret += f"\\draw[-latexslim] ({new_start[0]:.1f},{new_start[1]:.1f}) to ({new_end[0]:.1f},{new_end[1]:.1f});\n"
                ret += f"\\node at ({new_mid_node[0]:.1f}, {new_mid_node[1]:.1f}) {{${labl}$}};\n"
                
            elif cur_mode in [CUR_MODE_3, CUR_MODE_4, CUR_MODE_5, CUR_MODE_6]:
                flow_dir = flow_direction[cur_mode-2]
                ret += f"\\draw ({x1:.1f},{y1:.1f}) to[cisource, f{flow_dir}=${labl}$] ({x2:.1f},{y2:.1f});\n"

            v_plot_extra = ""
            if not meas_comp_same_direction:
                x1, y1, x2, y2 = x2, y2, x1, y1
                v_plot_extra = "^"
            if measure_type == MEAS_TYPE_VOLTAGE:
                ret += f"\\ctikzset{{american}};\n\\draw ({x1:.1f},{y1:.1f}) to[open, v{v_plot_extra}=${measure_label}$] ({x2:.1f},{y2:.1f});\n\\ctikzset{{european}};\n"
                
            return ret

    elif style == "american":
        pass

    elif style == "european":

        pass

    else:
        raise NotImplementedError
    pass



# NOTE: SPICE formatting
SPICE_TEMPLATES = {
    "v4": ".title Active DC Circuit {id}\n{components}\n.END\n",
    "v5": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v6": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v7": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v8": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v9": ".title Active DC Circuit\n{components}\n\n{simulation}.END\n",
    "v10": ".title Active DC Circuit\n{components}\n\n{simulation}.end\n",
    "v11": ".title Active DC Circuit\n{components}\n\n{simulation}.end\n",
}
SPICE_PREFFIX = {
    TYPE_RESISTOR: "R",
    TYPE_CAPACITOR: "C",
    TYPE_INDUCTOR: "L",
    TYPE_VOLTAGE_SOURCE: "V",
    TYPE_CURRENT_SOURCE: "I",
    TYPE_VCCS: "G",
    TYPE_VCVS: "E",
    TYPE_CCCS: "F",
    TYPE_CCVS: "H",
    TYPE_OPEN: "",
    TYPE_SHORT: "",
}

class Circuit:

    def __init__(self, m = 3, n = 4, 
                vertical_dis = None, horizontal_dis = None,
                has_vedge = None, has_hedge = None,

                vcomp_type = None, hcomp_type = None,
                vcomp_label = None, hcomp_label = None,             # only support string label
                vcomp_value = None, hcomp_value = None,
                vcomp_value_unit = None, hcomp_value_unit = None,
                vcomp_direction = None, hcomp_direction = None,

                vcomp_measure = None, hcomp_measure = None,
                vcomp_measure_label = None, hcomp_measure_label = None,     # only support string label
                vcomp_measure_direction = None, hcomp_measure_direction = None,

                vcomp_control_meas_label = None, hcomp_control_meas_label = None,     # only support string label  ==> For Controlled Source

                use_value_annotation = False,
                note = "v1",
                id = "",
                label_numerical_subscript = True,
                ):
        self.m = m
        self.n = n
        self.vertical_dis = np.arange(m)*4.0 if vertical_dis is None else vertical_dis
        self.horizontal_dis = np.arange(n)*3.0 if horizontal_dis is None else horizontal_dis

        self.has_vedge = np.ones((m-1, n)) if has_vedge is None else has_vedge # 1 or 0
        self.has_hedge = np.ones((m, n-1)) if has_hedge is None else has_hedge

        self.vcomp_type = np.zeros((m-1, n)) if vcomp_type is None else vcomp_type
        self.hcomp_type = np.zeros((m, n-1)) if hcomp_type is None else hcomp_type
        self.vcomp_label = np.ones((m-1, n)) if vcomp_label is None else vcomp_label
        self.hcomp_label = np.ones((m, n-1)) if hcomp_label is None else hcomp_label
        self.vcomp_value = np.zeros((m-1, n)) if vcomp_value is None else vcomp_value
        self.hcomp_value = np.zeros((m, n-1)) if hcomp_value is None else hcomp_value
        self.vcomp_value_unit = np.zeros((m-1, n)) if vcomp_value_unit is None else vcomp_value_unit
        self.hcomp_value_unit = np.zeros((m, n-1)) if hcomp_value_unit is None else hcomp_value_unit

        self.vcomp_direction = np.zeros((m-1, n)) if vcomp_direction is None else vcomp_direction # 0: n1==>n2, 1: n2==>n1
        self.hcomp_direction = np.zeros((m, n-1)) if hcomp_direction is None else hcomp_direction # 0: n1==>n2, 1: n2==>n1

        self.vcomp_measure = np.zeros((m-1, n)) if vcomp_measure is None else vcomp_measure
        self.hcomp_measure = np.zeros((m, n-1)) if hcomp_measure is None else hcomp_measure

        self.vcomp_measure_label = np.zeros((m-1, n)) if vcomp_measure_label is None else vcomp_measure_label
        self.hcomp_measure_label = np.zeros((m, n-1)) if hcomp_measure_label is None else hcomp_measure_label

        self.vcomp_measure_direction = np.zeros((m-1, n)) if vcomp_measure_direction is None else vcomp_measure_direction
        self.hcomp_measure_direction = np.zeros((m, n-1)) if hcomp_measure_direction is None else hcomp_measure_direction

        self.vcomp_control_meas_label = np.zeros((m-1, n)) if vcomp_control_meas_label is None else vcomp_control_meas_label
        self.hcomp_control_meas_label = np.zeros((m, n-1)) if hcomp_control_meas_label is None else hcomp_control_meas_label

        self.use_value_annotation = use_value_annotation # MEAS: True: annotate value in the figure / False: annotate label in the figure

        self.latex_font_size = "\\large"

        if self.use_value_annotation:
            self.label_numerical_subscript = True
        else:
            self.label_numerical_subscript = label_numerical_subscript

        
        self.note = note
        self.id = id

        self._init_degree() # initialize degree
        self._check_circuit_valid_by_degree() # check if the circuit is valid via degree
        self._init_netlist() # init netlist, and check if the circuit is valid by the topology

    def _init_degree(self):
        self.degree = np.zeros((self.m, self.n))

        for i in range(self.m):
            for j in range(self.n):
                if j>0:
                    self.degree[i][j] += (self.has_hedge[i][j-1] and self.hcomp_type[i][j-1] != TYPE_OPEN)
                if j<self.n-1:
                    self.degree[i][j] += (self.has_hedge[i][j] and self.hcomp_type[i][j] != TYPE_OPEN)
                if i>0:
                    self.degree[i][j] += (self.has_vedge[i-1][j] and self.vcomp_type[i-1][j] != TYPE_OPEN)
                if i<self.m-1:
                    self.degree[i][j] += (self.has_vedge[i][j] and self.vcomp_type[i][j] != TYPE_OPEN)

        self._degree_init = True
    
    def _get_grid_nodes(self):
        m, n = self.m, self.n
        visited = [[False]*n for _ in range(m)]  # Track visited nodes
        components = []  # Store the connected components
        
        def dfs(i, j, component):
            if i < 0 or i >= m or j < 0 or j >= n or visited[i][j]:
                return
            visited[i][j] = True
            component.append((i, j))
            
            # Traverse vedges and hedges
            if i > 0 and self.has_vedge[i-1][j] and self.vcomp_type[i-1][j] == TYPE_SHORT and self.vcomp_measure[i-1][j] == MEAS_TYPE_NONE: dfs(i-1, j, component)
            if j > 0 and self.has_hedge[i][j-1] and self.hcomp_type[i][j-1] == TYPE_SHORT and self.hcomp_measure[i][j-1] == MEAS_TYPE_NONE: dfs(i, j-1, component)
            if i < m-1 and self.has_vedge[i][j] and self.vcomp_type[i][j] == TYPE_SHORT and self.vcomp_measure[i][j] == MEAS_TYPE_NONE: dfs(i+1, j, component)
            if j < n-1 and self.has_hedge[i][j] and self.hcomp_type[i][j] == TYPE_SHORT and self.hcomp_measure[i][j] == MEAS_TYPE_NONE: dfs(i, j+1, component)
            
        for i in range(m):
            for j in range(n):
                if not visited[i][j]: 
                    component = []
                    dfs(i, j, component)
                    components.append(component)

        print(f"components: {components}")
        self.nodes = [f"{i}" for i in range(len(components))]
        grid_nodes = np.zeros((m, n)) # 0 by default
        for i in range(len(components)):
            for x, y in components[i]:
                grid_nodes[x][y] = i
        
        return grid_nodes
    
    def _check_conflict_component_measure(self, comp_type, comp_measure):
        conflict_pairs = [
            (TYPE_SHORT, MEAS_TYPE_VOLTAGE),
            (TYPE_OPEN, MEAS_TYPE_CURRENT),
            (TYPE_VOLTAGE_SOURCE, MEAS_TYPE_VOLTAGE),
            (TYPE_VCVS, MEAS_TYPE_VOLTAGE),
            (TYPE_CCVS, MEAS_TYPE_VOLTAGE),
            (TYPE_CURRENT_SOURCE, MEAS_TYPE_CURRENT),
            (TYPE_VCCS, MEAS_TYPE_CURRENT),
            (TYPE_CCCS, MEAS_TYPE_CURRENT),
        ]
        for pair in conflict_pairs:
            if comp_type == pair[0] and comp_measure == pair[1]:
                return True
        return False

    def init_netlist(self):
        return self._init_netlist()

    def _init_netlist(self):
        """
            Nodes: [Node1, Node2, ...]
            Branch: {Node1, Node2, type, lable, value, info}
        """
        assert self._degree_init, "degree not initialized"

        self.nodes = []
        self.branches = []
        
        self.grid_nodes = self._get_grid_nodes()
        print(f"Grid Nodes:\n{self.grid_nodes}\n\n")

        print("self.hcomp_type: \n", self.hcomp_type)
        print("self.has_hedge: \n", self.has_hedge)

        add_order = 0
        for i in range(self.m):
            for j in range(self.n):
                if int(self.note[1:]) <= 9:
                    raise NotImplementedError
                elif int(self.note[1:]) > 9:
                    print(f"({i}, {j}) / ({self.m}, {self.n})")
                    if j < self.n-1 and self.has_hedge[i][j]:
                        print(f"({i}, {j}) has hedge")
                        assert self.hcomp_type[i][j] != TYPE_OPEN, f"open circuit should not be in the netlist, {self.hcomp_type[i][j]}"
                        if self.grid_nodes[i][j] == self.grid_nodes[i][j+1]:
                            if self.hcomp_type[i][j] != TYPE_SHORT:
                                print("invalid circuit, some components are shorted")
                                self.valid = False
                                return False
                        
                        else:
                            n1 = f"{int(self.grid_nodes[i][j])}"
                            n2 = f"{int(self.grid_nodes[i][j+1])}"
                            if self.hcomp_direction[i][j]:
                                n1, n2 = n2, n1
                            new_branch = {
                                "n1": n1,
                                "n2": n2,
                                "type": self.hcomp_type[i][j],
                                "label": self.hcomp_label[i][j],
                                "value": self.hcomp_value[i][j],
                                "value_unit": self.hcomp_value_unit[i][j],
                                "measure": self.hcomp_measure[i][j],
                                "measure_label": self.hcomp_measure_label[i][j],
                                "meas_comp_same_direction": self.hcomp_measure_direction[i][j] == self.hcomp_direction[i][j],
                                "control_measure_label": self.hcomp_control_meas_label[i][j],
                                "info": "",
                                "order": add_order
                            }

                            if i == 1 and j == 1:
                                print(f"new_branch: {new_branch} on [1, 1]")

                            if self._check_conflict_component_measure(self.hcomp_type[i][j], self.hcomp_measure[i][j]):
                                print("invalid circuit, conflict between component type and measure type")
                                self.valid = False
                                return False
                        
                            self.branches.append(new_branch)
                            add_order += 1
                    
                    if i < self.m-1 and self.has_vedge[i][j]:
                        print(f"({i}, {j}) has vedge")
                        if self.grid_nodes[i][j] == self.grid_nodes[i+1][j]:
                            if self.vcomp_type[i][j] != TYPE_SHORT:
                                print("invalid circuit, some components are shorted")
                                self.valid = False
                                return False
                            
                        else:   # 不等价节点的边
                            n1 = f"{int(self.grid_nodes[i][j])}"
                            n2 = f"{int(self.grid_nodes[i+1][j])}"
                            if self.vcomp_direction[i][j]:
                                n1, n2 = n2, n1
                            new_branch = {
                                "n1": n1,
                                "n2": n2,
                                "type": self.vcomp_type[i][j],
                                "label": self.vcomp_label[i][j],
                                "value": self.vcomp_value[i][j],
                                "value_unit": self.vcomp_value_unit[i][j],
                                "measure": self.vcomp_measure[i][j],
                                "measure_label": self.vcomp_measure_label[i][j],
                                "meas_comp_same_direction": self.vcomp_measure_direction[i][j] == self.vcomp_direction[i][j],
                                "control_measure_label": self.vcomp_control_meas_label[i][j],
                                "info": "",
                                "order": add_order
                            }

                            if self._check_conflict_component_measure(self.vcomp_type[i][j], self.vcomp_measure[i][j]):
                                print("invalid circuit, conflict between component type and measure type")
                                self.valid = False
                                return False

                            self.branches.append(new_branch)
                            add_order += 1

        for br in self.branches:
            tmp = [(b['n1'], b['n2']) for b in self.branches if b['measure_label'] == br['control_measure_label'] and b['measure'] == MEAS_TYPE_VOLTAGE]
            if len(tmp) != 1:
                print(f"Controlled Source should have one and only one voltage measurement, but got {len(tmp)}, {br['control_measure_label']}")
                self.valid = False
                return False
        
        # TODO: check if the graph is invalid: has two voltage in parallel or two current in series
                            
        print(f"init netlist done, get branches: {self.branches}")
        return True
        pass
    
    def _to_SPICE(self):
        """
        example in OP:
        .title Active DC Circuit
        R1 1 2 4k
        R2 3 2 4k
        R3 1 NR3 2k
        VI NR3 0 0
        R4 3 0 3k
        VS1 1 3 25
        IS1 3 2 3m
        IS2 0 1 10m
        IS3 0 2 5m

        .control
        op
        print I(vi)
        * print v(1,2)
        .endc
        .end
        """
        spice_str = ""

        # NOTE: Element Card
        if int(self.note[1:]) <= 9:
            raise NotImplementedError
        elif int(self.note[1:]) > 9:
            sorted_branches = sorted(self.branches, key=lambda x: x["order"])
            for br in sorted_branches:
                meas_comp_same_direction = br["meas_comp_same_direction"]
                ms_label_str = "" if br["measure_label"] == -1 else str(int(br["measure_label"]))
                ctr_ms_label_str = "" if br["control_measure_label"] == -1 else str(int(br["control_measure_label"])) 

                value_write = str(int(br["value"]))+unit_scales[br["value_unit"]] if self.use_value_annotation else "<Empty>"
                label_write = "" if self.use_value_annotation else str(br["label"])
                # NOTE: For value annotation, the label is not annotated in SPICE;
                #       For label annotation, the value is not annotated in SPICE;

                print(br["type"], br["label"], br["n1"], br["n2"], br["value"], br["value_unit"])
                type_str = SPICE_PREFFIX[br['type']]

                if br["type"] == TYPE_SHORT:
                    assert br["measure"] == MEAS_TYPE_CURRENT, f"short circuit should be measured by current, {br}"
                    vmeas_str = f"VI{ms_label_str}"
                    spice_str += "%s %s %s %s\n" % (vmeas_str, br["n1"], br["n2"], 0)
                
                if br["type"] in [TYPE_VOLTAGE_SOURCE, TYPE_CURRENT_SOURCE, TYPE_RESISTOR]:
                    if br["measure"] == MEAS_TYPE_CURRENT:
                        mid_node = "N%s%s" % (br['n1'], br['n2'])
                        vmeas_str = f"VI{ms_label_str}"
                        spice_str += "%s%s %s %s %s\n" %  (type_str, label_write,  br["n1"],   mid_node,   value_write)
                        spice_str += "%s %s %s 0\n" %       (vmeas_str,             mid_node,   br["n2"]) if meas_comp_same_direction \
                                else "%s %s %s 0\n" % (vmeas_str, br["n2"], mid_node)
                    else:
                        spice_str += "%s%s %s %s %s\n" %   (type_str, label_write,  br["n1"],   br["n2"],   value_write)

                if br["type"] in [TYPE_CCVS, TYPE_CCCS]:    # 流控电压源、流控电流源

                    tmp = [b for b in self.branches if b['measure_label'] == br['control_measure_label'] and b['measure'] == MEAS_TYPE_CURRENT]
                    assert len(tmp) == 1, "Controlled Source should have one and only one voltage measurement, but got %d, %d" % (len(tmp), br['control_measure_label'])

                    control_measure_str = f"VI{ctr_ms_label_str}"

                    if br["measure"] == MEAS_TYPE_CURRENT:
                        mid_node = "N%s%s" % (br['n1'], br['n2'])
                        vmeas_str = f"VI{ms_label_str}"
                        spice_str += "%s%s %s %s %s %s\n" %  (type_str, label_write,  br["n1"],   mid_node,   control_measure_str,  value_write)
                        spice_str += "%s %s %s 0\n" %       (vmeas_str,             mid_node,   br["n2"]) if meas_comp_same_direction \
                                else "%s %s %s 0\n" % (vmeas_str, br["n2"], mid_node)
                    else:
                        spice_str += "%s%s %s %s %s %s\n" %   (type_str, label_write,  br["n1"],   br["n2"],  control_measure_str,   value_write)
            
                if br["type"] in [TYPE_VCVS, TYPE_VCCS]:    # 压控电压源、压控电流源

                    tmp = [(b['n1'], b['n2']) for b in self.branches if b['measure_label'] == br['control_measure_label'] and b['measure'] == MEAS_TYPE_VOLTAGE]
                    assert len(tmp) == 1, "Controlled Source should have one and only one voltage measurement, but got %d, %d" % (len(tmp), br['control_measure_label'])

                    control_n1, control_n2 = tmp[0]

                    if br["measure"] == MEAS_TYPE_CURRENT:
                        mid_node = "N%s%s" % (br['n1'], br['n2'])
                        vmeas_str = f"VI{ms_label_str}"
                        spice_str += "%s%s %s %s %s %s %s\n" %  (type_str, label_write,  br["n1"],   mid_node,   control_n1,  control_n2,  value_write)
                        spice_str += "%s %s %s 0\n" %       (vmeas_str,             mid_node,   br["n2"]) if meas_comp_same_direction \
                                else "%s %s %s 0\n" % (vmeas_str, br["n2"], mid_node)
                    else:
                        spice_str += "%s%s %s %s %s %s %s\n" %  (type_str, label_write,  br["n1"],   br["n2"],   control_n1,  control_n2,  value_write)

        # NOTE: Control Card
        if int(self.note[1:]) <= 9:
            raise NotImplementedError
        elif int(self.note[1:]) > 9:
            zero_order = True
            for br in self.branches:
                if br["type"] in [TYPE_CAPACITOR, TYPE_INDUCTOR]:
                    zero_order = False
                    break

            if zero_order:      # 零阶电路
                sim_str = ".control\nop\n"
                for br in self.branches:
                    if br["measure_label"] == -1:
                        ms_label_str = ""
                    else:
                        ms_label_str = str(int(br["measure_label"]))

                    if br["measure"] == MEAS_TYPE_VOLTAGE:
                        print(f"#n1: {br['n1']}, n2: {br['n2']}")
                        # sim_str += f".PRINT DC V({br['n1']}, {br['n2']}) * measurement of U{br['measure_label']}\n"
                        meas_n1, meas_n2 = br["n1"], br["n2"]
                        if not br["meas_comp_same_direction"]:
                            meas_n1, meas_n2 = meas_n2, meas_n1
                        if str(meas_n1) == '0':
                            sim_str += "print -v(%s) ; measurement of U%s\n" % (meas_n2, ms_label_str)
                        elif str(meas_n2) == '0':
                            sim_str += "print v(%s) ; measurement of U%s\n" % (meas_n1, ms_label_str)
                        else:
                            sim_str += "print v(%s, %s) ; measurement of U%s\n" % (meas_n1, meas_n2, ms_label_str)
                    elif br["measure"] == MEAS_TYPE_CURRENT:
                        print('#')
                        # sim_str += f".PRINT DC V({br['n1']}, {br['n2']}) / (R{br['label']}) * measurement of I{br['measure_label']} : I(R{br['label']})\n"
                        vmeas_str = f"VI{ms_label_str}"
                        sim_str += "print i(%s) ; measurement of I%s\n" % (vmeas_str, ms_label_str)
                sim_str += ".endc\n"
                print(f"spice_str: {spice_str}, \n\nsim_str: {sim_str}\n\n")
                # exit()
                spice_str = SPICE_TEMPLATES[self.note].format(components=spice_str, simulation=sim_str)   

            else:   # high order circuit
                # TODO: add transient simulation
                raise NotImplementedError        
        else:
            raise NotImplementedError

        return spice_str
        pass

    def _check_circuit_valid_by_degree(self):

        assert self._degree_init, "degree not initialized"

        # check if the degree is valid (all not equal to 1)
        self.valid = True
        for i in range(self.m):
            for j in range(self.n):
                if self.degree[i][j] == 1:
                    print("invalid cricuit")
                    self.valid = False
        
        # TODO: check if there are voltage source in parallel OR current source in series
        if self.valid:
            print("valid circuit")
        else:
            print("invalid circuit")

    def _draw_vertical_edge(self, i, j):
        if ((i>=0 and i<self.m-1) and (j>=0 and j<self.n)) and self.has_vedge[i][j]:
            if int(self.note[1:]) < 9: # <= version 4
                raise NotImplementedError
            else:
                new_line = get_latex_line_draw(self.horizontal_dis[j], self.vertical_dis[i], self.horizontal_dis[j], self.vertical_dis[i+1],
                                                self.vcomp_type[i][j], 
                                                self.vcomp_label[i][j], 
                                                self.vcomp_value[i][j], 
                                                self.vcomp_value_unit[i][j],
                                                self.use_value_annotation,
                                                measure_type=self.vcomp_measure[i][j], 
                                                measure_label=self.vcomp_measure_label[i][j],
                                                measure_direction=self.vcomp_measure_direction[i][j],
                                                direction=self.vcomp_direction[i][j],
                                                label_subscript_type=int(not self.label_numerical_subscript),
                                                control_label=self.vcomp_control_meas_label[i][j],
                                                note=self.note
                                            )
            return new_line
        else:
            return ""
        
    def _draw_horizontal_edge(self, i, j):
        if ((i>=0 and i<self.m) and (j>=0 and j<self.n-1)) and self.has_hedge[i][j]:
            if int(self.note[1:]) < 9: # <= version 4
                raise NotImplementedError
            else:
                new_line = get_latex_line_draw(self.horizontal_dis[j], self.vertical_dis[i], self.horizontal_dis[j+1], self.vertical_dis[i],
                                                self.hcomp_type[i][j], 
                                                self.hcomp_label[i][j], 
                                                self.hcomp_value[i][j],
                                                self.hcomp_value_unit[i][j],
                                                self.use_value_annotation,
                                                measure_type=self.hcomp_measure[i][j], 
                                                measure_label=self.hcomp_measure_label[i][j],
                                                measure_direction=self.hcomp_measure_direction[i][j],
                                                direction=self.hcomp_direction[i][j],
                                                label_subscript_type=int(not self.label_numerical_subscript),
                                                control_label=self.hcomp_control_meas_label[i][j],
                                                note=self.note
                                            )
            return new_line
        else: 
            return ""
        
    def to_latex(self):
        # with open("./templates/latex_template.txt", "r") as f:
        #     latex_template = f.read()
        if int(self.note[1:]) <= 9:
            raise NotImplementedError
        elif int(self.note[1:]) > 9:
            latex_template = LATEX_TEMPLATES["v9"]
        else:
            raise NotImplementedError
        
        latex_code_main = ""
        for i in range(self.m):
            for j in range(self.n):
                latex_code_main += self._draw_horizontal_edge(i,j)
                latex_code_main += self._draw_vertical_edge(i,j)
        latex_code = latex_template.replace("<main>", latex_code_main)
        
        if int(self.note[1:]) >= 8:
            latex_code = latex_code.replace("<font>", self.latex_font_size)

        return latex_code

def gen_circuit(note="v1", id=""):

    ## v1-9 old version
    if int(note[1:]) <= 9:
        raise NotImplementedError
    
    ## v10
    elif int(note[1:]) == 10:

        num_edge_choices = [2]*3 + [3]*5 + [4]*3 + [5]*2 + [6]*1 + [7]*1 + [8]*1
        num_source_choices = [TYPE_VOLTAGE_SOURCE]*5 + [TYPE_CURRENT_SOURCE]*5 + [TYPE_VCCS]*2 + [TYPE_VCVS]*2 + [TYPE_CCCS]*2 + [TYPE_CCVS]*2

        m = np.random.choice(num_edge_choices)
        # n = 3 + np.random.randint(-1, 3)
        n = np.random.choice(num_edge_choices)
        vertical_dis = np.arange(m)* 3 + np.random.uniform(-0.5, 0.5, size=(m,))
        horizontal_dis = np.arange(n)* 3 + np.random.uniform(-0.5, 0.5, size=(n,))

        num_short_max = 0
        # cut_outer_edge_rate = 0.8
        cut_outer_edge_rate = 1
        cut_corner_rate = 0.2

        cut_left_top = random.random()<cut_corner_rate
        cut_left_bottom = random.random()<cut_corner_rate
        cut_right_top = random.random()<cut_corner_rate
        cut_right_bottom = random.random()<cut_corner_rate
        while num_short_max < 1:
            has_vedge = np.random.randint(0, 2, size=(m-1, n)) # 0 or 1
            has_hedge = np.random.randint(0, 2, size=(m, n-1)) # 0 or 1

            for i in range(m-1):
                has_vedge[i][0] = int(random.random() < cut_outer_edge_rate)
                has_vedge[i][n-1] = int(random.random() < cut_outer_edge_rate)
            # has_vedge[:, [0,n-1]] = 1; # left and right
            has_hedge = np.random.randint(0, 2, size=(m, n-1)) # 0 or 1
            for j in range(n-1):
                has_hedge[0][j] = int(random.random() < cut_outer_edge_rate)
                has_hedge[m-1][j] = int(random.random() < cut_outer_edge_rate)
            # has_hedge[[0,m-1], :] = 1; # top and bottom
            
            num_edges = np.sum(has_vedge) + np.sum(has_hedge)
            if num_edges > 8:
                if cut_left_bottom:
                    has_vedge[0][0] = 0
                    has_hedge[0][0] = 0
                if cut_left_top:
                    has_vedge[m-2][0] = 0
                    has_hedge[m-1][0] = 0
                if cut_right_bottom:
                    has_vedge[0][n-1] = 0
                    has_hedge[0][n-2] = 0
                if cut_right_top:
                    has_vedge[m-2][n-1] = 0
                    has_hedge[m-1][n-2] = 0

            idxs_has_vedge = np.where(has_vedge == 1)
            idxs_has_vedge = list(zip(idxs_has_vedge[0], idxs_has_vedge[1]))
            idxs_has_hedge = np.where(has_hedge == 1)
            idxs_has_hedge = list(zip(idxs_has_hedge[0], idxs_has_hedge[1]))
            idxs_edge = [(0, i, j) for i, j in idxs_has_vedge] + [(1, i, j) for i, j in idxs_has_hedge]
            
            num_edges = len(idxs_has_vedge) + len(idxs_has_hedge)
            max_num_source = max(min(5, num_edges // 2 - 1), 1)
            num_sources = np.random.randint(1, max_num_source+1)
            sources = np.random.choice(num_source_choices, num_sources)

            num_volsrs = np.sum(sources == TYPE_VOLTAGE_SOURCE)
            num_cursrs = np.sum(sources == TYPE_CURRENT_SOURCE)
            num_vccs = np.sum(sources == TYPE_VCCS)
            num_vcvs = np.sum(sources == TYPE_VCVS)
            num_cccs = np.sum(sources == TYPE_CCCS)
            num_ccvs = np.sum(sources == TYPE_CCVS)

            num_short_max = (num_edges - num_sources) - 2

        print(f"num_short_max: {num_short_max}")
        print(idxs_edge)
        num_short = np.random.randint(0, num_short_max+1)
        num_open = np.random.randint(0, num_short // 2) if num_short > 2 else 0
        num_r = num_edges - num_sources - num_short  - num_open# Resistor

        np.random.shuffle(idxs_edge)
        idxs_volsrc = idxs_edge[:num_volsrs]
        idxs_cursrc = idxs_edge[num_volsrs:num_volsrs+num_cursrs]
        idxs_vccs = idxs_edge[num_volsrs+num_cursrs:num_volsrs+num_cursrs+num_vccs]
        idxs_vcvs = idxs_edge[num_volsrs+num_cursrs+num_vccs:num_volsrs+num_cursrs+num_vccs+num_vcvs]
        idxs_cccs = idxs_edge[num_volsrs+num_cursrs+num_vccs+num_vcvs:num_volsrs+num_cursrs+num_vccs+num_vcvs+num_cccs]
        idxs_ccvs = idxs_edge[num_volsrs+num_cursrs+num_vccs+num_vcvs+num_cccs:num_sources]
        idxs_r = idxs_edge[num_sources:num_sources+num_r]
        idxs_open = idxs_edge[num_sources+num_r:num_sources+num_r+num_open]

        label_volsrc = np.random.permutation(range(num_volsrs)) + 1
        label_cursrc = np.random.permutation(range(num_cursrs)) + 1
        label_vccs = np.random.permutation(range(num_vccs)) + 1
        label_vcvs = np.random.permutation(range(num_vcvs)) + 1
        label_cccs = np.random.permutation(range(num_cccs)) + 1
        label_ccvs = np.random.permutation(range(num_ccvs)) + 1

        label_r = np.random.permutation(range(num_r)) + 1

        vcomp_type = np.zeros((m-1, n))
        hcomp_type = np.zeros((m, n-1))
        vcomp_label = np.zeros((m-1, n))
        hcomp_label = np.zeros((m, n-1))
        vcomp_value = np.zeros((m-1, n))
        hcomp_value = np.zeros((m, n-1))

        vcomp_value_unit = np.zeros((m-1, n))
        hcomp_value_unit = np.zeros((m, n-1))

        vcomp_direction = np.random.randint(0, 2, size=(m-1, n)) # 0 or 1
        hcomp_direction = np.random.randint(0, 2, size=(m, n-1)) # 0 or 1

        vcomp_measure = np.zeros((m-1, n))
        hcomp_measure = np.zeros((m, n-1))

        vcomp_measure_label = np.zeros((m-1, n))
        hcomp_measure_label = np.zeros((m, n-1))

        vcomp_measure_direction = np.random.randint(0, 2, size=(m-1, n)) # 0 or 1
        hcomp_measure_direction = np.random.randint(0, 2, size=(m, n-1)) # 0 or 1

        vcomp_control_meas_label = np.zeros((m-1, n))   
        hcomp_control_meas_label = np.zeros((m, n-1))
        
        min_value_r, max_value_r = 1, 100
        min_value_v, max_value_v = 1, 100
        min_value_i, max_value_i = 1, 100

        # add measuremaent
        num_measure_choices = list(range(0, num_r+1)) + [0]*5+[1]*5+[2]*2
        num_measure = np.random.choice(num_measure_choices)
        if num_measure > 0:
            num_measure_i = np.random.randint(0, num_measure+1)
            num_measure_v = num_measure - num_measure_i
        else:
            num_measure_i = 0
            num_measure_v = 0
        if num_measure_i < num_cccs + num_ccvs:
            num_measure_i = num_cccs + num_ccvs
        if num_measure_v < num_vccs + num_vcvs:
            num_measure_v = num_vccs + num_vcvs
        num_measure = num_measure_i + num_measure_v

        measure_label_sets = np.random.choice(range(-1, 100), num_measure, replace=False)
        
        idxs_measure_i = random.sample(idxs_edge, num_measure_i)
        idxs_measure_v = random.sample(list(set(idxs_edge) - set(idxs_measure_i)) + (idxs_cursrc), num_measure_v)
    
        for l, (s, i, j) in enumerate(idxs_measure_i):
            if s == 0:
                vcomp_measure[i][j] = MEAS_TYPE_CURRENT
                vcomp_measure_label[i][j] = measure_label_sets[l]
            else:
                hcomp_measure[i][j] = MEAS_TYPE_CURRENT
                hcomp_measure_label[i][j] = measure_label_sets[l]
        for l, (s, i, j) in enumerate(idxs_measure_v):
            if s == 0:
                vcomp_measure[i][j] = MEAS_TYPE_VOLTAGE
                vcomp_measure_label[i][j] = measure_label_sets[l]
            else:
                hcomp_measure[i][j] = MEAS_TYPE_VOLTAGE
                hcomp_measure_label[i][j] = measure_label_sets[l]


        for l, (s, i, j) in enumerate(idxs_open):
            if s == 0:
                vcomp_type[i][j] = TYPE_OPEN
            else:
                hcomp_type[i][j] = TYPE_OPEN

        for l, (s, i, j) in enumerate(idxs_volsrc):
            if s == 0:
                vcomp_type[i][j] = TYPE_VOLTAGE_SOURCE
                vcomp_label[i][j] = label_volsrc[l]
                vcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
            else:
                hcomp_type[i][j] = TYPE_VOLTAGE_SOURCE
                hcomp_label[i][j] = label_volsrc[l]
                hcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
        for l, (s, i, j) in enumerate(idxs_cursrc):
            if s == 0:
                vcomp_type[i][j] = TYPE_CURRENT_SOURCE
                vcomp_label[i][j] = label_cursrc[l]
                vcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
            else:
                hcomp_type[i][j] = TYPE_CURRENT_SOURCE
                hcomp_label[i][j] = label_cursrc[l]
                hcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
        for l, (s, i, j) in enumerate(idxs_r):
            if s == 0:
                vcomp_type[i][j] = TYPE_RESISTOR
                vcomp_label[i][j] = label_r[l]
                vcomp_value[i][j] = np.random.randint(min_value_r, max_value_r)
            else:
                hcomp_type[i][j] = TYPE_RESISTOR
                hcomp_label[i][j] = label_r[l]
                hcomp_value[i][j] = np.random.randint(min_value_r, max_value_r)
        for l, (s, i, j) in enumerate(idxs_vccs):
            if s == 0:
                vcomp_type[i][j] = TYPE_VCCS
                vcomp_label[i][j] = label_vccs[l]
                vcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)       
            else:
                hcomp_type[i][j] = TYPE_VCCS
                hcomp_label[i][j] = label_vccs[l]
                hcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
        for l, (s, i, j) in enumerate(idxs_vcvs):
            if s == 0:
                vcomp_type[i][j] = TYPE_VCVS
                vcomp_label[i][j] = label_vcvs[l]
                vcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
            else:
                hcomp_type[i][j] = TYPE_VCVS
                hcomp_label[i][j] = label_vcvs[l]
                hcomp_value[i][j] = np.random.randint(min_value_v, max_value_v)
        for l, (s, i, j) in enumerate(idxs_cccs):
            if s == 0:
                vcomp_type[i][j] = TYPE_CCCS
                vcomp_label[i][j] = label_cccs[l]
                vcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
            else:
                hcomp_type[i][j] = TYPE_CCCS
                hcomp_label[i][j] = label_cccs[l]
                hcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
        for l, (s, i, j) in enumerate(idxs_ccvs):
            if s == 0:
                vcomp_type[i][j] = TYPE_CCVS
                vcomp_label[i][j] = label_ccvs[l]
                vcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)
            else:
                hcomp_type[i][j] = TYPE_CCVS
                hcomp_label[i][j] = label_ccvs[l]
                hcomp_value[i][j] = np.random.randint(min_value_i, max_value_i)

        # 添加控制源
        for l, (s,i,j) in enumerate(idxs_vccs + idxs_vcvs):
            if s == 0:
                control_measure_voltage_idx = random.choice(idxs_measure_v)
                if control_measure_voltage_idx[0] == 0:
                    vcomp_control_meas_label[i][j] = vcomp_measure_label[control_measure_voltage_idx[1]][control_measure_voltage_idx[2]]
                elif control_measure_voltage_idx[0] == 1:
                    vcomp_control_meas_label[i][j] = hcomp_measure_label[control_measure_voltage_idx[1]][control_measure_voltage_idx[2]]
            else:
                control_measure_voltage_idx = random.choice(idxs_measure_v)
                if control_measure_voltage_idx[0] == 0:
                    hcomp_control_meas_label[i][j] = vcomp_measure_label[control_measure_voltage_idx[1]][control_measure_voltage_idx[2]]
                elif control_measure_voltage_idx[0] == 1:
                    hcomp_control_meas_label[i][j] = hcomp_measure_label[control_measure_voltage_idx[1]][control_measure_voltage_idx[2]]
        for l, (s,i,j) in enumerate(idxs_cccs + idxs_ccvs):
            if s == 0:
                control_measure_current_idx = random.choice(idxs_measure_i)
                if control_measure_current_idx[0] == 0:
                    vcomp_control_meas_label[i][j] = vcomp_measure_label[control_measure_current_idx[1]][control_measure_current_idx[2]]
                elif control_measure_current_idx[0] == 1:
                    vcomp_control_meas_label[i][j] = hcomp_measure_label[control_measure_current_idx[1]][control_measure_current_idx[2]]
            else:
                control_measure_current_idx = random.choice(idxs_measure_i)
                if control_measure_current_idx[0] == 0:
                    hcomp_control_meas_label[i][j] = vcomp_measure_label[control_measure_current_idx[1]][control_measure_current_idx[2]]
                elif control_measure_current_idx[0] == 1:
                    hcomp_control_meas_label[i][j] = hcomp_measure_label[control_measure_current_idx[1]][control_measure_current_idx[2]]

        print(f"vcomp_value: {vcomp_value}\n\nhcomp_value: {hcomp_value}")
        print(f"vcomp_value_unit: {vcomp_value_unit}\n\nhcomp_value_unit: {hcomp_value_unit}")

        unit_choices = [UNIT_MODE_1]*10 + [UNIT_MODE_k]*4 + [UNIT_MODE_m]*2
        vcomp_value_unit = np.random.choice(unit_choices, size=(m-1, n))
        hcomp_value_unit = np.random.choice(unit_choices, size=(m, n-1))
        
        # use_value_annotation = False
        use_value_annotation = bool(random.getrandbits(1))
        # label_str_subscript = bool(random.getrandbits(1)) & ~use_value_annotation
        label_str_subscript = False
        label_numerical_subscript = not label_str_subscript

        # Convert all matrix to int
        vcomp_type = vcomp_type.astype(int)
        hcomp_type = hcomp_type.astype(int)
        vcomp_label = vcomp_label.astype(int)
        hcomp_label = hcomp_label.astype(int)
        vcomp_value = vcomp_value.astype(int)
        hcomp_value = hcomp_value.astype(int)
        vcomp_value_unit = vcomp_value_unit.astype(int)
        hcomp_value_unit = hcomp_value_unit.astype(int)

        vcomp_measure = vcomp_measure.astype(int)
        hcomp_measure = hcomp_measure.astype(int)
        vcomp_measure_label = vcomp_measure_label.astype(int)
        hcomp_measure_label = hcomp_measure_label.astype(int)
        vcomp_measure_direction = vcomp_measure_direction.astype(int)
        hcomp_measure_direction = hcomp_measure_direction.astype(int)
        vcomp_control_meas_label = vcomp_control_meas_label.astype(int)
        hcomp_control_meas_label = hcomp_control_meas_label.astype(int)

        print("#"*100)
        print("Generate a random grid for circuit ... ")
        print(f"has_vedge: {has_vedge}\n\nhas_hedge: {has_hedge}")
        print(f"vertical_dis: {vertical_dis}\n\nhorizontal_dis: {horizontal_dis}")
        print(f"m:{m}, n:{n}\n\nnum_edges:{num_edges},\nnum_sources: {num_sources},\nnum_volsrs: {num_volsrs},\nnum_cursrs: {num_cursrs}\nnum_resistors: {num_r}")
        print(f"use_value_annotation: {use_value_annotation}\nlabel_numerical_subscript: {label_numerical_subscript}")

        print(f"vcomp_type: {vcomp_type}\n\nhcomp_type: {hcomp_type}")
        print(f"vcomp_label: {vcomp_label}\n\nhcomp_label: {hcomp_label}")
        print(f"vcomp_value: {vcomp_value}\n\nhcomp_value: {hcomp_value}")
        print(f"vcomp_value_unit: {vcomp_value_unit}\n\nhcomp_value_unit: {hcomp_value_unit}")
        print(f"vcomp_measure: {vcomp_measure}\n\nhcomp_measure: {hcomp_measure}")
        print(f"vcomp_measure_label: {vcomp_measure_label}\n\nhcomp_measure_label: {hcomp_measure_label}")
        print(f"vcomp_measure_direction: {vcomp_measure_direction}\n\nhcomp_measure_direction: {hcomp_measure_direction}")
        print(f"vcomp_control_meas_label: {vcomp_control_meas_label}\n\nhcomp_control_meas_label: {hcomp_control_meas_label}")

        # print(f"Generating a circuit grid of size {m}x{n} with {num_volsrs} voltage sources, {num_cursrs} current sources, and {num_r} resistors.")
        circ = Circuit( m=m, n=n, \
                        vertical_dis=vertical_dis, horizontal_dis=horizontal_dis, \
                        has_vedge=has_vedge, has_hedge=has_hedge, \
                        vcomp_type=vcomp_type, hcomp_type=hcomp_type, \
                        vcomp_label=vcomp_label, hcomp_label=hcomp_label, \
                        vcomp_value=vcomp_value, hcomp_value=hcomp_value, \
                        vcomp_value_unit=vcomp_value_unit, hcomp_value_unit=hcomp_value_unit, \
                        vcomp_measure=vcomp_measure, hcomp_measure=hcomp_measure, \
                        vcomp_measure_label=vcomp_measure_label, hcomp_measure_label=hcomp_measure_label, \
                        use_value_annotation=use_value_annotation, note=note, id=id,
                        vcomp_direction=vcomp_direction, hcomp_direction=hcomp_direction,
                        vcomp_measure_direction=vcomp_measure_direction, hcomp_measure_direction=hcomp_measure_direction,
                        vcomp_control_meas_label=vcomp_control_meas_label, hcomp_control_meas_label=hcomp_control_meas_label,
                        label_numerical_subscript=label_numerical_subscript)    # whether use numerical subscript for label
    
    elif int(note[1:]) == 11:

        # Set distribution & Hyperparameters
        num_grid_options = [2, 3, 4, 5, 6, 7, 8]
        num_grid_dis = [3, 6, 6, 2, 1, 0, 0]
        num_grid_choices = []
        for op, dis in zip(num_grid_options, num_grid_dis):
            num_grid_choices += [op]*dis
 
        num_comp_dis = [10, 5, 5, 20, 0, 0, 5, 2, 2, 2, 2]  # Short, V, I, R, C, L, Open, VCCS, VCVS, CCCS, CCVS
        num_comp_dis_outer = [10, 5, 5, 20, 0, 0, 0, 2, 2, 2, 2]    # in the outer loop: no <open>
        num_comp_choices = []
        num_comp_choices_outer = []
        for op, dis in zip(range(11), num_comp_dis):
            num_comp_choices += [op]*dis
        for op, dis in zip(range(11), num_comp_dis_outer):
            num_comp_choices_outer += [op]*dis
        
        vertical_dis_mean, vertical_dis_std = 3, 0.5
        horizontal_dis_mean, horizontal_dis_std = 3, 0.5

        comp_mean_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        comp_max_value = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

        unit_dis = [10, 4, 2]
        unit_choices = [UNIT_MODE_1]*unit_dis[0] + [UNIT_MODE_k]*unit_dis[1] + [UNIT_MODE_m]*unit_dis[2]

        meas_dis = [10, 1, 1]
        meas_choices = [MEAS_TYPE_NONE]*meas_dis[0] + [MEAS_TYPE_VOLTAGE]*meas_dis[1] + [MEAS_TYPE_CURRENT]*meas_dis[2]
        meas_dir_prob = 0.5

        meas_label_choices = range(-1, 10)

        use_value_annotation_prob = 0.8

        # Get the grid
        m = np.random.choice(num_grid_choices)
        n = np.random.choice(num_grid_choices)
        vertical_dis = np.arange(m)* vertical_dis_mean + np.random.uniform(-vertical_dis_std, vertical_dis_std, size=(m,))
        horizontal_dis = np.arange(n)* horizontal_dis_mean + np.random.uniform(-horizontal_dis_std, horizontal_dis_std, size=(n,))

        while True:

            # Get the number of edges
            has_vedge = np.ones((m-1, n), dtype=int)
            has_hedge = np.ones((m, n-1), dtype=int)

            vcomp_type = np.zeros((m-1, n), dtype=int)
            hcomp_type = np.zeros((m, n-1), dtype=int)
            vcomp_label = np.zeros((m-1, n))
            hcomp_label = np.zeros((m, n-1))
            vcomp_value = np.zeros((m-1, n))
            hcomp_value = np.zeros((m, n-1))

            vcomp_value_unit = np.zeros((m-1, n), dtype=int)
            hcomp_value_unit = np.zeros((m, n-1), dtype=int)

            vcomp_direction = np.zeros((m-1, n), dtype=int) # 0 or 1
            hcomp_direction = np.zeros((m, n-1), dtype=int) # 0 or 1

            vcomp_measure = np.zeros((m-1, n), dtype=int)
            hcomp_measure = np.zeros((m, n-1), dtype=int)

            vcomp_measure_label = np.zeros((m-1, n))
            hcomp_measure_label = np.zeros((m, n-1))

            vcomp_measure_direction = np.zeros((m-1, n), dtype=int) # 0 or 1
            hcomp_measure_direction = np.zeros((m, n-1), dtype=int) # 0 or 1

            vcomp_control_meas_label = np.zeros((m-1, n))   
            hcomp_control_meas_label = np.zeros((m, n-1))

            # Get the components
            comp_cnt = [0] * 11
            meas_label_stat = {
                MEAS_TYPE_NONE: [],
                MEAS_TYPE_VOLTAGE: [],
                MEAS_TYPE_CURRENT: []
            }

            ## type, value, value_unit, label
            VC_sources = {'v': [], 'h': []}
            IC_sources = {'v': [], 'h': []}
            print(f"has_vedge: {has_vedge}\n\nhas_hedge: {has_hedge}")

            for i in range(m-1):
                for j in range(n):
                    if j == 0 or j == n-1:
                        vcomp_type[i][j] = np.random.choice(num_comp_choices_outer)
                    else:
                        vcomp_type[i][j] = np.random.choice(num_comp_choices)

                    if vcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS]:
                        VC_sources["v"].append((i, j))
                    if vcomp_type[i][j] in [TYPE_CCCS, TYPE_CCVS]:
                        IC_sources["v"].append((i, j))
                    if vcomp_type[i][j] == TYPE_OPEN:
                        has_vedge[i][j] = 0
                        continue

                    vcomp_value[i][j] = np.random.randint(comp_mean_value[vcomp_type[i][j]], comp_max_value[vcomp_type[i][j]])
                    vcomp_value_unit[i][j] = np.random.choice(unit_choices)

                    comp_cnt[vcomp_type[i][j]] += 1
                    vcomp_label[i][j] = comp_cnt[vcomp_type[i][j]]

                    vcomp_measure[i][j] = np.random.choice(meas_choices)
                    vcomp_measure_label[i][j] = np.random.choice(meas_label_choices)
                    meas_label_stat[vcomp_measure[i][j]].append(vcomp_measure_label[i][j])
                    vcomp_direction[i][j] = int(random.random() < meas_dir_prob)

                    print(f"\n\nvcomp_type[{i}][{j}]: {vcomp_type[i][j]}, vcomp_value[{i}][{j}]: {vcomp_value[i][j]}, vcomp_value_unit[{i}][{j}]: {vcomp_value_unit[i][j]}")
                    print(f"vcomp_measure[{i}][{j}]: {vcomp_measure[i][j]}, vcomp_measure_label[{i}][{j}]: {vcomp_measure_label[i][j]}, vcomp_direction[{i}][{j}]: {vcomp_direction[i][j]}")
            for i in range(m):
                for j in range(n-1):
                    if i == 0 or i == m-1:
                        hcomp_type[i][j] = np.random.choice(num_comp_choices_outer)
                    else:
                        hcomp_type[i][j] = np.random.choice(num_comp_choices)

                    if hcomp_type[i][j] in [TYPE_VCCS, TYPE_VCVS]:
                        VC_sources["h"].append((i, j))
                    if hcomp_type[i][j] in [TYPE_CCCS, TYPE_CCVS]:
                        IC_sources["h"].append((i, j))
                    if hcomp_type[i][j] == TYPE_OPEN:
                        has_hedge[i][j] = 0
                        continue
                    
                    hcomp_value[i][j] = np.random.randint(comp_mean_value[hcomp_type[i][j]], comp_max_value[hcomp_type[i][j]])
                    hcomp_value_unit[i][j] = np.random.choice(unit_choices)

                    comp_cnt[hcomp_type[i][j]] += 1
                    hcomp_label[i][j] = comp_cnt[hcomp_type[i][j]]

                    hcomp_measure[i][j] = np.random.choice(meas_choices)
                    hcomp_measure_label[i][j] = np.random.choice(meas_label_choices)
                    meas_label_stat[hcomp_measure[i][j]].append(hcomp_measure_label[i][j])
                    hcomp_direction[i][j] = int(random.random() < meas_dir_prob)

                    print(f"\n\nhcomp_type[{i}][{j}]: {hcomp_type[i][j]}, hcomp_value[{i}][{j}]: {hcomp_value[i][j]}, hcomp_value_unit[{i}][{j}]: {hcomp_value_unit[i][j]}")
                    print(f"hcomp_measure[{i}][{j}]: {hcomp_measure[i][j]}, hcomp_measure_label[{i}][{j}]: {hcomp_measure_label[i][j]}, hcomp_direction[{i}][{j}]: {hcomp_direction[i][j]}")
            
            # Check the control source
            num_vc_sources = len(VC_sources["v"]) + len(VC_sources["h"])
            num_ic_sources = len(IC_sources["v"]) + len(IC_sources["h"])
            num_vmeas = len(meas_label_stat[MEAS_TYPE_VOLTAGE])
            num_imeas = len(meas_label_stat[MEAS_TYPE_CURRENT])

            if (num_vc_sources > 0 and num_vmeas == 0) or (num_ic_sources > 0 and num_imeas == 0):
                continue

            print("VC_sources: ", VC_sources)
            print("IC_sources: ", IC_sources)
            print("meas_label_stat: ", meas_label_stat)

            for i, j in VC_sources["v"]:
                contrl_idx = random.choice(meas_label_stat[MEAS_TYPE_VOLTAGE])
                vcomp_control_meas_label[i][j] = contrl_idx
            for i, j in VC_sources["h"]:
                contrl_idx = random.choice(meas_label_stat[MEAS_TYPE_VOLTAGE])
                hcomp_control_meas_label[i][j] = contrl_idx
            for i, j in IC_sources["v"]:
                contrl_idx = random.choice(meas_label_stat[MEAS_TYPE_CURRENT])
                vcomp_control_meas_label[i][j] = contrl_idx
            for i, j in IC_sources["h"]:
                contrl_idx = random.choice(meas_label_stat[MEAS_TYPE_CURRENT])
                hcomp_control_meas_label[i][j] = contrl_idx
            break
        
        # use_value_annotation = False
        use_value_annotation = bool(random.random() < use_value_annotation_prob)
        # label_str_subscript = bool(random.getrandbits(1)) & ~use_value_annotation
        label_str_subscript = False
        label_numerical_subscript = not label_str_subscript

        # Convert all matrix to int
        vcomp_type = vcomp_type.astype(int)
        hcomp_type = hcomp_type.astype(int)
        vcomp_label = vcomp_label.astype(int)
        hcomp_label = hcomp_label.astype(int)
        vcomp_value = vcomp_value.astype(int)
        hcomp_value = hcomp_value.astype(int)
        vcomp_value_unit = vcomp_value_unit.astype(int)
        hcomp_value_unit = hcomp_value_unit.astype(int)
        vcomp_measure = vcomp_measure.astype(int)
        hcomp_measure = hcomp_measure.astype(int)
        vcomp_measure_label = vcomp_measure_label.astype(int)
        hcomp_measure_label = hcomp_measure_label.astype(int)
        vcomp_measure_direction = vcomp_measure_direction.astype(int)
        hcomp_measure_direction = hcomp_measure_direction.astype(int)
        vcomp_control_meas_label = vcomp_control_meas_label.astype(int)
        hcomp_control_meas_label = hcomp_control_meas_label.astype(int)

        print("#"*100)
        print("Generate a random grid for circuit ... ")
        print(f"has_vedge: {has_vedge}\n\nhas_hedge: {has_hedge}")
        print(f"vertical_dis: {vertical_dis}\n\nhorizontal_dis: {horizontal_dis}")
        print(f"m:{m}, n:{n}\n\ncomp_cnt: {json.dumps(comp_cnt, indent=4)}")
        print(f"use_value_annotation: {use_value_annotation}\nlabel_numerical_subscript: {label_numerical_subscript}")

        print(f"vcomp_type: {vcomp_type}\n\nhcomp_type: {hcomp_type}")
        print(f"vcomp_label: {vcomp_label}\n\nhcomp_label: {hcomp_label}")
        print(f"vcomp_value: {vcomp_value}\n\nhcomp_value: {hcomp_value}")
        print(f"vcomp_value_unit: {vcomp_value_unit}\n\nhcomp_value_unit: {hcomp_value_unit}")
        print(f"vcomp_measure: {vcomp_measure}\n\nhcomp_measure: {hcomp_measure}")
        print(f"vcomp_measure_label: {vcomp_measure_label}\n\nhcomp_measure_label: {hcomp_measure_label}")
        print(f"vcomp_measure_direction: {vcomp_measure_direction}\n\nhcomp_measure_direction: {hcomp_measure_direction}")
        print(f"vcomp_control_meas_label: {vcomp_control_meas_label}\n\nhcomp_control_meas_label: {hcomp_control_meas_label}")

        # print(f"Generating a circuit grid of size {m}x{n} with {num_volsrs} voltage sources, {num_cursrs} current sources, and {num_r} resistors.")
        circ = Circuit(m, n, vertical_dis, horizontal_dis, has_vedge, has_hedge, vcomp_type, hcomp_type, vcomp_label, hcomp_label, \
                        vcomp_value=vcomp_value, hcomp_value=hcomp_value, \
                        vcomp_value_unit=vcomp_value_unit, hcomp_value_unit=hcomp_value_unit, \
                        vcomp_measure=vcomp_measure, hcomp_measure=hcomp_measure, \
                        vcomp_measure_label=vcomp_measure_label, hcomp_measure_label=hcomp_measure_label, \
                        use_value_annotation=use_value_annotation, note=note, id=id,
                        vcomp_direction=vcomp_direction, hcomp_direction=hcomp_direction,
                        vcomp_measure_direction=vcomp_measure_direction, hcomp_measure_direction=hcomp_measure_direction,
                        vcomp_control_meas_label=vcomp_control_meas_label, hcomp_control_meas_label=hcomp_control_meas_label,
                        label_numerical_subscript=label_numerical_subscript)


    else:
        circ = Circuit()
    return circ
