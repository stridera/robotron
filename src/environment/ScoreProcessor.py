import numpy as np
import hashlib
import base64


class ScoreProcessor():
    SCORE_BOX = np.array([479, 91])
    SCORE_SIZE = np.array([12, 14])
    SCORE_BUFFER = 2

    NUMBERS_HASH = {
        'wyj8ay':0,'warr4l':0,'bbbnw5':0,'trkkxr':0,'1jiwqc':0,'ceh3iu':0,'r4swfw':0,'3dbr5o':0,'dqos4u':0,'kx9cgf':0,'g5c/iv':0,'k95ozf':0,'xdywgt':0,'4ezgmb':0,
        'tldks7':1,'szcrhd':1,'wf5wgw':1,'9etfwr':1,'ykoeur':1,'zfctly':1,'xhsqw2':1,'tnoezf':1,'cewbyp':1,'n/z8cd':1,'d+qvoc':1,'+ugtdk':1,
        'rht93c':2,'042urk':2,'grbo36':2,'c90lxp':2,'5j34un':2,'ohcpnt':2,'ujhlc7':2,'qulngk':2,'vrvp7j':2,'eerjiq':2,'fydmir':2,'wi8rxd':2,'cdyngv':2,'f751x+':2,'/w4kld':2,
        'vwujcu':3,'oqca4r':3,'jefo8a':3,'7jjfwq':3,'vbpjip':3,'+etzhn':3,'c/+zoa':3,'j4a5z5':3,'e4badr':3,'u2f8cd':3,
        'yskjgx':4,'vmjz8y':4,'pvased':4,'lpssgx':4,'ijtxus':4,'vmjz8y':4,'pvased':4,'lpssgx':4,'bgcvif':4,'2zmxyl':4,'kndv5o':4,'8dbv8b':4,'hsgm9s':4,
        'ssop8b':5,'nbr42g':5,'b2ngl9':5,'t6x5lz':5,'1jveha':5,'zt2j2m':5,'xrloea':5,'bosspd':5,'u7sc9o':5,'ocvril':5,'s//eqz':5,'4vz+wc':5,'+jpqpv':5,'hvctsc':5,'sl4ogs':5,'i6zcip':5,'44a0cs':5,'fqrmvx':5,'dxbmqj':5,
        'bynaqu':6,'aj2mo7':6,'obnnen':6,'xftabr':6,'fldafz':6,
        'jst1wf':7,'vhfsgy':7,'49f4x9':7,'fg06b2':7,'5ms0tb':7,'tsfu0q':7,'ttmg5e':7,'8weloy':7,'3hen02':7,'ryupe5':7,'ot26gj':7,'rignx4':7,'x/pesb':7,
        'sqyncb':8,'bvirlr':8,'g559ta':8,'hpn6iu':8,'ednnrk':8,
        'tosdv5':9,'ronwec':9,'sldlii':9,'clibho':9,'dj6rbu':9,'jhnpc2':9,

        # catch non-numbers... perhaps we can use these to know when we're transitioning
        'zhxyap':-1,'90dd+e':-1,'/ao8j8':-1,'ywgemb':-1,'mpejnl':-1,'gccexm':-1,'jly+ic':-1,'8dribl':-1,'fxazsm':-1,'+ieq3f':-1,'scd6pq':-1,'kme+xy':-1,'vdp0zy':-1,'at2zis':-1,'fl0/fk':-1,'rzcihe':-1,'nyfcla':-1,'r/r6fa':-1,'swemgn':-1,'elv4lf':-1,'sx8zxm':-1,'jjahka':-1,'ynpx+1':-1,'+3zmlu':-1,'/jaxop':-1,'wkyk9e':-1,'jsyya6':-1,'vekwkn':-1,'gg62jt':-1,'/71uxu':-1,'ydtn2e':-1,'osmjkf':-1,'0kyeh6':-1,'jndem2':-1,'1+zyj2':-1,'s3down':-1,'ms5h7e':-1,'404njd':-1,'r1vdh+':-1,'vi+jsn':-1,'/j4kco':-1,'hq/v6g':-1,'llpnnx':-1,'xbkfod':-1
    }

    def __init__(self):
        ''' constructor '''

    def h6(self, w):
        h = hashlib.md5(w).digest()
        return base64.b64encode(h)[:6].lower().decode("utf-8")

    def numberCleanup(self, image):
        binary_output = np.zeros_like(image)
        binary_output[image >= 30] = 1
        return binary_output

    def getDigitFromImageBinary(self, score_bin):
        digest = self.h6(score_bin)

        if digest in self.NUMBERS_HASH:
            return self.NUMBERS_HASH[digest]

        print(type(digest), type(list(self.NUMBERS_HASH.keys())[0]))
        print("Missing Hash", digest, score_bin)
        return -1

    def getScore(self, gray):
        score = -1
        tl = np.copy(self.SCORE_BOX)
        br = tl + self.SCORE_SIZE
        score_img = gray[tl[1]:br[1], tl[0]:br[0]]
        score_bin = self.numberCleanup(score_img)

        place = 0
        while (np.count_nonzero(score_bin)):
            digit = self.getDigitFromImageBinary(score_bin)
            if digit == -1:
                print("digit", digit, "place", place, "score", score)
                return -1

            if place == 0:
                score = digit
            else:
                score += digit * (10 ** place)

            tl[0] = tl[0] - self.SCORE_SIZE[0] - self.SCORE_BUFFER
            br[0] = tl[0] + self.SCORE_SIZE[0]
            score_img = gray[tl[1]:br[1], tl[0]:br[0]]
            score_bin = self.numberCleanup(score_img)
            place += 1

        return score
