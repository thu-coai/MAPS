# MIT License
#
# Copyright (c) 2024 Diordany van Hemert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys as m_sys
import os as m_os
class RawFile:
  def __init__(self, p_fileName):
    self.file = None
    self.fileName = p_fileName

    self.title = None
    self.dataStr = None
    self.plotName = None
    self.simFlags = None
    self.varCount = None
    self.dataCount = None

    self.vars = {}

  def check_header_marker(self, p_label):
    marker = p_label+":"

    line = self.file.readline().strip()

    if not line == marker:
      m_sys.exit("Marker '"+p_label+"' not found.")

  def close(self):
    self.file.close()

  def get_curve_data(self, p_vars):
    data = []

    nVars = len(p_vars)

    if nVars > 0:
      for varName in p_vars:
        if varName in self.vars:
          if self.vars[varName]["type"] != "time":
            data.append({"name": varName, "type": self.vars[varName]["type"], "data": self.vars[varName]["data"]})
    else:
      for varName in self.vars:
        if self.vars[varName]["type"] != "time":
          data.append({"name": varName, "type": self.vars[varName]["type"], "data": self.vars[varName]["data"]})

    return data

  def get_time_data(self):
    for varName in self.vars:
      if (varName == "time") and (self.vars[varName]["type"] == "time"):
        return self.vars[varName]["data"]

    m_sys.exit("Couldn't find the time variable.")

  def open(self):
    if not m_os.path.isfile(self.fileName):
      m_sys.exit("The file '"+self.fileName+"' does not exist.")

    self.file = open(self.fileName, "r")

  def read(self):
    self.read_header()
    self.read_data()

  def read_data(self):
    self.check_header_marker("Values")

    for i in range(self.dataCount):
      for varName in self.vars:
        if (varName == "time") and (self.vars[varName]["type"] == "time"):
          prefix = str(i)+"\t\t"
          self.vars[varName]["data"].append(float(self.file.readline().strip()[len(prefix):]))
        else:
          self.vars[varName]["data"].append(float(self.file.readline().strip()))

  def read_header(self):
    self.title = self.read_header_line("Title")
    self.dateStr = self.read_header_line("Date")
    self.plotName = self.read_header_line("Plotname")
    self.simFlags = self.read_header_line("Flags")
    self.varCount = int(self.read_header_line("No. Variables"))
    self.dataCount = int(self.read_header_line("No. Points"))
    self.read_header_vars()

  def read_header_line(self, p_label):
    prefix = p_label+": "

    line = self.file.readline().strip()

    if not line.startswith(prefix):
      m_sys.exit("Line with label '"+p_label+"' not found.")

    return line[len(prefix):]

  def read_header_vars(self):
    self.check_header_marker("Variables")

    for i in range(self.varCount):
      var = self.file.readline().strip().split("\t")
      self.vars[var[1]] = {"type": var[2], "data": []}