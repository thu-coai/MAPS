import matplotlib.pyplot as plt
import os, argparse, re, json
from rawfile import RawFile

TMP_FILE = "__tmp.sp"


def transient_analysis(config):
    os.system("ngspice -b -r test.raw " + config.circuit)
    rawFile = RawFile("test.raw")
    rawFile.open()
    rawFile.read()
    rawFile.close()
    os.remove("test.raw")

    time = rawFile.get_time_data()
    vars = []
    with open(config.circuit, "r", encoding="utf-8") as file:
        pattern = re.compile(r"plot\s+(.*)")
        for line in file:
            match = pattern.match(line)
            if match:
                vars.append(match.group(1).lower())
    curves = rawFile.get_curve_data(vars)

    if len(curves) > 0:
        figure, axes = plt.subplots(len(curves))
        figure.suptitle(rawFile.title + " [" + rawFile.plotName + "]")

        xlabel = "Time / s"
        ylabel = None
        if len(curves) > 1:
            for i, curve in enumerate(curves):
                ylabel = curve["name"]
                scale = max(abs(max(curve["data"])), abs(min(curve["data"])))
                if scale > 2e5:
                    axes[i].plot(time, [x / 1e6 for x in curve["data"]])
                    ylabel += " / M"
                elif scale > 2e2:
                    axes[i].plot(time, [x / 1e3 for x in curve["data"]])
                    ylabel += " / k"
                elif scale > 2e-1:
                    axes[i].plot(time, curve["data"])
                    ylabel += " / "
                elif scale > 2e-4:
                    axes[i].plot(time, [x * 1e3 for x in curve["data"]])
                    ylabel += " / m"
                elif scale > 2e-7:
                    axes[i].plot(time, [x * 1e6 for x in curve["data"]])
                    ylabel += " / μ"

                if curve["name"][0] == "v":
                    ylabel += "V"
                elif curve["name"][0] == "i":
                    ylabel += "A"

                axes[i].set_ylabel(ylabel)
                axes[i].set_xlabel("Time / s")
                axes[i].grid()
        else:
            ylabel = curves[0]["name"]
            scale = max(abs(max(curves[0]["data"])), abs(min(curves[0]["data"])))
            if scale > 2e5:
                plt.plot(time, [x / 1e6 for x in curves[0]["data"]])
                ylabel += " / M"
            elif scale > 2e2:
                plt.plot(time, [x / 1e3 for x in curves[0]["data"]])
                ylabel += " / k"
            elif scale > 2e-1:
                plt.plot(time, curves[0]["data"])
                ylabel += " / "
            elif scale > 2e-4:
                plt.plot(time, [x * 1e3 for x in curves[0]["data"]])
                ylabel += " / m"
            elif scale > 2e-7:
                plt.plot(time, [x * 1e6 for x in curves[0]["data"]])
                ylabel += " / μ"

            if curves[0]["name"][0] == "v":
                ylabel += "V"
            elif curves[0]["name"][0] == "i":
                ylabel += "A"

            plt.ylabel(ylabel)
            plt.xlabel("Time / s")
            plt.grid()

        if config.save:
            rawFile.title = re.sub(r"[\/\\\:\*\?\"\<\>\|]", "", rawFile.title)
            plt.savefig(rawFile.title + ".png")
        else:
            plt.show()
    else:
        print("No curves to show.")


def op_analysis(config):
    user_require_variables = []
    with open(config.circuit, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if line.lower().startswith(".print") or line.lower().startswith("print"):
                match = re.match(r"print\s+(.*)", line.lower())
                if match:
                    user_require_variables += match.group(1).split()

    simvar2userquery = {}
    with open(TMP_FILE, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line)
            if line.lower().startswith(".op") or line.lower().startswith("op"):
                file.write("\nprint allv\n")
            elif ";" in line:
                # comment = line.split(";")[1]
                print(line)
                match = re.match(r"print\s+([\w\(\)\-\s,]*)\s+;\s+measurement\s+of\s+(\w*)", line)
                print(match)
                if match:
                    simvar, userquery = match.group(1).lower().replace(' ', ''), match.group(2).lower()
                    simvar2userquery[simvar] = userquery
    print(f"\n\nSimVar to UserQuery: {simvar2userquery}\n\n")
    config.circuit = TMP_FILE

    os.system("ngspice -b -o output.txt " + config.circuit)

    result = {}
    if not os.path.exists("output.txt"):
        print("No output file found.")
        return
    
    with open("output.txt", "r", encoding="utf-8") as file:
        result["raw_file"] = file.read()
    with open("output.txt", "r", encoding="utf-8") as file:
        for line in file:
            # 如果这一行的格式是aaa = bbb，其中aaa是包含字母、数字、下划线、圆括号的字符串，bbb由数字、小数点、e、+、-组成
            if re.match(r"^[\w\(\)\-,\s]+\s*=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", line):
                # print(line, end="")
                match = re.match(r"^([\w\(\)\-,\s]+)\s*=\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$", line)
                if match:
                    print(match.group(0), match.group(1), match.group(2), match.group(3))
                    variable = match.group(1).strip().lower().replace(' ', '')
                    value = float(match.group(2))

                    print(f"variable: {variable}, value: {value}\n")
                    if value.is_integer():
                        value = int(value)
                    # 如果variable被i(xx)或v(xx)包围，去掉括号和括号前面的i/v
                    # print(f"variable: {variable}, userquery: {simvar2userquery[variable]}")
                    if variable in simvar2userquery.keys():
                        # result[("I" if variable[0] == "i" else "U") + str(user_require_variables.index(variable))] = (
                        result[ simvar2userquery[variable] ] = (
                            str(value) + ("A" if variable[0] == "i" else "V")
                        )
                    else:
                        match = re.match(r"([iv])\((.*)\)", variable)
                        if match:
                            variable = match.group(2)
                            result[variable] = str(value) + ("A" if match.group(1) == "i" else "V")
                        else:
                            result[variable] = str(value) + "V"

    # os.remove("output.txt")

    with open(config.output, "w", encoding="utf-8") as file:
        file.write(json.dumps(result, indent=4))


TYPE = None

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--circuit", help="Circuit file")
    argParser.add_argument("-s", "--spice", help="Spice")
    argParser.add_argument("--save", help="Save plot to file", action="store_true")
    argParser.add_argument("-o", "--output", help="Output file", default="output.json")

    config = argParser.parse_args()
    if config.spice:
        config.spice = config.spice.replace("\\n", "\n")
        with open(TMP_FILE, "w", encoding="utf-8") as file:
            file.write(config.spice)
        config.circuit = TMP_FILE

    with open(config.circuit, "r", encoding="utf-8") as file:
        for line in file:
            if line.lower().startswith(".tran"):
                TYPE = "TRAN"
                break
            if line.lower().startswith(".op") or line.lower().startswith("op"):
                TYPE = "OP"
                break

    if TYPE == "TRAN":
        transient_analysis(config)
    elif TYPE == "OP":
        op_analysis(config)
    else:
        print("No analysis found.")

    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)
