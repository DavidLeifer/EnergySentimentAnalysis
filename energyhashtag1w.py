# this script is the same as the other one
#   except for the access_ and consumer_ keys and tokens
# Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from datetime import datetime as dt
import os
import sys
import logging
import time
import subprocess

python_program = r'C:\Users\siu854530344\Downloads\drive-download-20180305T174920Z-001\energyhashtag1w.py'
the_other_python_script = r'C:\Users\siu854530344\Downloads\drive-download-20180305T174920Z-001\energyhashtag2w.py'

logfilename = 'energy' + dt.now().strftime("%Y%m%d%H%M%S") + '.log'
logging.basicConfig(filename=logfilename, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

# Variables that contains the user credentials to access Twitter API 
access_token = "XXX"
access_token_secret = "XXX"
consumer_key = "XXX"
consumer_secret = "XXX"


# This is a basic listener that writes received tweets to files.
# Each file contains max 10000 tweets
class MyListener(StreamListener):
    MAINFILENAME = 'energy'
    MAXTWEETSINFILE = 10000

    def __init__(self, api=None):
        self.api = api
        super(StreamListener, self).__init__()
        self.tweets_count = 0
        self.current_file = self.get_file()
        self.previous_file = self.current_file
        self.rate_limit_exceeded = False
        
    def on_data(self, data):
        self.tweets_count += 1
        self.current_file.write(str(data))
        if self.tweets_count > MyListener.MAXTWEETSINFILE:
            self.previous_file.close()
            self.current_file = self.get_file()
            self.tweets_count = 0
        return True

    def on_exception(self, exception):
        print "on_exception in energyhashtag1w.py on " + dt.now().strftime("%Y%m%d%H%M%S")
        logging.warn('on_exception in energyhashtag1w.py')
        self.running = False
        subprocess.call([sys.executable, python_program, the_other_python_script])

    def on_error(self, status):
        print "on_error in energyhashtag1w.py on " + dt.now().strftime("%Y%m%d%H%M%S")
        self.current_file.close()
        if status == 420:
            logging.warn('rate limit exceeded.')
            self.rate_limit_exceeded = True
        logging.warn('on_error in energyhashtag1w.py with status code ' + str(status))
        self.running = False
        subprocess.call([sys.executable, python_program, the_other_python_script])

    def get_file(self):
        name = MyListener.MAINFILENAME + dt.now().strftime("%Y%m%dT%H%M%S")
        name += '.txt'
        return open(name, 'a')

def stop_stream(stream):
    stream.listener.stop()
    stream.disconnect()
    stream.listener.current_file.close()

if __name__ == '__main__':

    stream = None
    try:
        #This handles Twitter authetification and the connection to
        # Twitter Streaming API
        l = MyListener()
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        stream = Stream(auth, l)

        #This line filter Twitter Streams to capture data by keywords
        stream.filter(track=['#gasoline','#Crudeoil', '#crude', '#oil',
'#OOTT', '#OPEC', '#algae', '#biodiesel', '#bioethanol', '#Biogas', '#biofuel',
'#biofuels', '#ethanol', '#biomass', '#AdvancedBiofuels', '#RFS', '#Fuelcell',
'#hydrogen', '#FossilEnergy', '#GreenEnergy', '#thermal', '#coal', '#coalmine',
'#geothermal', '#hydroenergy', '#MethaneHydrates', '#biomethane', '#biopower',
'#natgas', '#naturalgas', '#SynthesisGas', '#nuclearenergy', '#nuclear',
'#OTEC', '#renewable', '#RenewableEnergy', '#TidalEnergy', '#WaveEnergy',
'#oceanenergy', '#shalegas', '#solarfarm', '#solar', '#SolarPower',
'#SolarEnergy', '#SolarPanels', '#WindEnergy', '#hydropower', '#hydroenergy',
'#windfarm', '#WindTurbine', '#RenewableNaturalGas', '#wastetofuel',
'#envirofuel', '#wastetoenergy', '#bioenergy', '#syngas', '#cellulosic',
'#gasoil', '#graphene', '#LPG', '#cleanenergy', '#Fracking', '#Gasification', '#syngas', '#THE'],async=True,stall_warnings=True)
    except:
        # when exceptions occur, the program did not go here
        logging.warning("main try except MyException")
        if stream is not None:
            stop_stream(stream)
        # hard-coded the path of the other script
        # os.execv('/home/dxiong/socialmedia/energyhashtag2.py', sys.argv)

