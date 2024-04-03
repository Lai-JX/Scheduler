from __future__ import print_function
import socket
import sys
import subprocess
import flags 
import logging
import math
import os
import xml.etree.ElementTree as ET
from contextlib import closing
import cvxpy as cp
import numpy as np

FLAGS = flags.FLAGS

def make_logger(name, path=None):
    LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
    print('log path:',path)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not path:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(ch)
    else:
        fh = logging.FileHandler(path)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(fh)

    return logger

def get_host_ip():
    """get the host ip elegantly
    https://www.chenyudong.com/archives/python-get-local-ip-graceful.html
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        # ljx: 由于我们使用的环境的ip是设置好的，详细看 /etc/hosts, 所以需要结合实际对应的ip改动一下
        # ip = ip.split('.')
        # ip[1],ip[2] = "0", "0"
        # ip[3] = str(int(ip[3]) + 9)
        # ip = ".".join(ip)
    finally:
        s.close()
    return ip

def print_fn(log):
    if FLAGS.print:
        print(log)
        if FLAGS.flush_stdout:
            sys.stdout.flush()


def print_ljx(*objects, sep=' ', end='\n', file=sys.stdout, flush=False):
    # print("current time:" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("output setted by LJX:", end="\n\t")
    print(*objects, sep=sep, end=end, file=file, flush=flush)



def mkdir(folder_path):
    cmd = 'mkdir -p ' + folder_path
    ret = subprocess.check_call(cmd, shell=True)
    print_fn(ret)


def search_dict_list(dict_list, key, value):
    '''
    Search the targeted <key, value> in the dict_list
    Return:
        list entry, or just None 
    '''
    for e in dict_list:
        # if e.has_key(key) == True:
        if key in e:
            if math.isclose(e[key], value, rel_tol=1e-9):
                return e

    return None

def parse_xml(filename:str):
    # print('parse_xml:',filename)
    fb_memory_usage = []
    utilization = []
    file_content = open(filename, mode='r').read()
    xmls = file_content.split('</nvidia_smi_log>\n')
    for i in range(len(xmls) - 1):
        root = ET.fromstring(xmls[i] + '</nvidia_smi_log>\n')
        for child in root[4]:
            if child.tag == 'fb_memory_usage':
                fb_memory_usage.append(child[2].text)   # ljx
            if child.tag == "utilization":
                utilization.append(child[0].text)
    return fb_memory_usage, utilization

def find_free_port():
    """
    https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    
# 算差分
def calDiff(data):
    data_diff = []
    for i in range(len(data)-1):
        data_diff.append((data[i+1]-data[i])/60)
    return data_diff


if __name__ == '__main__':
    print(get_host_ip())
    print(print_ljx("hhh"))
    print(parse_xml("../ljx.xml"))

    data = [106095.7938488306, 106371.08212895904, 106502.54001553723, 106831.1847319827, 107226.71831424593, 107304.04648282133,107342.71056710904, 107570.4420235636, 107827.55818407684, 108000.0]
    print(calDiff(data))