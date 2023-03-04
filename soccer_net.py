import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader


mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="/media/josh/TURVER/soccer_net_vidoes/")

mySoccerNetDownloader.password = input("s0cc3rn3t")

mySoccerNetDownloader.downloadDataTask(task="reid-2023", split=["train", "valid", "test", "challenge"])

# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])