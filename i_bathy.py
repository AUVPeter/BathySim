from datetime import datetime
import time
import pymoos
from bathy_sim import BathySim

class iBathy:
  

  def __init__(self):

    self._x = 0.0
    self._y = 0.0
    self._z = 0.0
    self._h = 0.0

    #Bathy Sim
    self.sim = BathySim(123)

    #MOOS
    self._moos = pymoos.comms()
    self._moos.set_on_connect_callback(self.on_connect)
    self._moos.set_on_mail_callback(self.on_mail)

    self._update = False
    self._logfile = None
    self._file_label = ""


  def trig_newfile(self):
    if self._logfile:
      self._logfile.close()
    time_string = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = time_string + self._file_label + '.bathy.txt'
    self._logfile = open(filename,'w')
    self._moos.notify('BATHY_FILENAME', filename)


  def handle_label(self, label):
    if label != self._file_label:
      self._file_label = label
      self.trig_newfile()

  def on_connect(self):
    ''' MOOS connection active callback '''
    self._moos.register('NAV_X')
    self._moos.register('NAV_Y')
    self._moos.register('NAV_DEPTH')
    self._moos.register('NAV_HEADING')
    self._moos.register('BATHY_LABEL')
    self.trig_newfile()
    return True


  def on_mail(self):
    ''' MOOS mail arrival callback '''
    msgs = self._moos.fetch()
    if len(msgs) > 0:
      self._update = True
    for msg in msgs:
      if msg.key() == 'NAV_X':
        self._x = msg.double()
      elif msg.key() == 'NAV_Y':
        self._y = msg.double()
      elif msg.key() == 'NAV_DEPTH':
        self._z = msg.double()
      elif msg.key() == 'NAV_HEADING':
        self._h = msg.double()
      elif msg.key() == 'BATHY_LABEL':
        print(msg.string())
        self.handle_label(msg.string())
    return True

 
  def run(self):
    ''' main program loop '''
    self._moos.run('localhost',9000,'iBathy')
    while True:
      if self._update:
        t = pymoos.time() 
        line = self.sim.mb_sample(self._x, self._y, self._h,depth=self._z,sensor=False)
        num = len(line[:,0])
        x_array = ",".join([f'{d:.2f}' for d in line[:,0]])
        y_array = ",".join([f'{d:.2f}' for d in line[:,1]])
        z_array = ",".join([f'{d:.2f}' for d in line[:,2]])
        self._moos.notify('BATHY_X', f"t={t:.2f},n={num},x={x_array}")
        self._moos.notify('BATHY_Y', f"t={t:.2f},n={num},y={y_array}")
        self._moos.notify('BATHY_Z', f"t={t:.2f},n={num},z={z_array}")

        self._logfile.write(f"iBATHY,{t:.2f},{num}\n")
        self._logfile.write(f"{x_array}\n{y_array}\n{z_array}\n")

        self._update = False
      time.sleep(.1) 


if __name__ == '__main__':
  app = iBathy()
  app.run()
