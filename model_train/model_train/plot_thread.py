import threading
import time
import matplotlib.pyplot as p
class plot_thread(threading.Thread):

    acc_list = None
    acc_list_moving_avg = None
    flag = False
    flag_end = False

    def set_param(self, acc_list, acc_list_moving_avg):
        self.flag = True
        self.acc_list = acc_list
        self.acc_list_moving_avg = acc_list_moving_avg

    def end_thread(self):
        self.flag_end = True

    def run(self):
        while self.flag == False:
            time.sleep(5)
            pass
        p.figure(1)
        p.ylim((0.3, 1))
        while self.flag_end == False:
            p.clf()
            p.plot(self.acc_list[0::10], 'b')
            p.plot(self.acc_list_moving_avg[1::10], 'r-')
            p.ylim((0.3, 1))
            #ax2.plot(loss_list, 'r')
            p.pause(0.001)
        p.savefig(r'E:\Visual Wake Words\script\model_train\model_train\acc.png',
                  format='png',dpi=300)
