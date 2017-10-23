import sys
sys.path.append("/home/machen/face_expr")
import psutil
from train_monitor.email_utils import EmailUtils
import time
import daemon
import argparse
import os

def get_pid_list(pid_folder):
    pid_list = []
    for pid_path in os.listdir(pid_folder):
        pid_path = pid_folder + os.sep + pid_path
        with open(pid_path, 'r') as file_obj:
            for pid in file_obj:
                pid = pid.strip()
                pid_list.append(pid)
    return pid_list

class Reporter(object):

    def report(self, title, info):
        EmailUtils.sendMail(["sharpstill@163.com"], title, info)


def monitor_pid_list(pid_folder, reporter, interval):

    has_report_pids = set()
    while True:
        pid_list = get_pid_list(pid_folder)
        all_title = []
        all_info = []
        for pid in pid_list:
            try:
                process = psutil.Process(pid=int(pid))
                with process.oneshot():
                    name = process.name()
                    cpu_times = process.cpu_times()  # accumulated process time, in seconds, (user, system, children_user, children_system)
                    elapsed_time = cpu_times[0] + cpu_times[1]
                    cpu_percent = process.cpu_percent(interval=None)  # cpu percentage since last call
                    status = process.status()
                    cmd_line = " ".join(process.cmdline())
                    if (status == psutil.STATUS_DEAD or status == psutil.STATUS_ZOMBIE) and pid not in has_report_pids:
                        has_report_pids.add(pid)
                        title = "pid:{0} {1}".format(pid, cmd_line)
                        info = "cmd line:{0} name:{1} status:{2} cpu_percent:{3} elapsed_time:{4} has hang(not running) since last call!".format(cmd_line,
                                                                                                                            name, status, cpu_percent,
                                                                                                                            elapsed_time)
                        all_title.append(title)
                        all_info.append(info)

            except psutil.ZombieProcess:
                title= "training process become ZombieProcess"
                info = "process pid:{} became zombie process".format(pid)
                if pid not in has_report_pids:
                    has_report_pids.add(pid)
                    reporter.report(title, info)
            except psutil.NoSuchProcess:
                title = "training process dead cannot find:{}".format(pid)
                info = "no such process pid:{}".format(pid)
                if pid not in has_report_pids:
                    has_report_pids.add(pid)
                    reporter.report(title, info)
        if all_title:
            all_title = ",".join(all_title)
            all_info = "<br />".join(all_info)
            reporter.report(all_title, all_info)
        time.sleep(interval) # each ten minutes to watch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='daemon process to watch process')
    parser.add_argument('--folder','-f', help="folder contains pid files")
    parser.add_argument('--interval',type=int,default=500, help='interval in seconds to watch')
    parser.add_argument('--pid', '-p', default='/tmp/watch_train.pid', help='watch dog itself pid file path')
    args = parser.parse_args()
    reporter = Reporter()
    with daemon.DaemonContext():
        pid = str(os.getpid())
        with open(args.pid, "w") as file_obj:
            file_obj.write(pid)
            file_obj.flush()
        monitor_pid_list(args.folder, reporter, args.interval)
