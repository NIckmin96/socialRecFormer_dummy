import logging

if __name__=='__main__':
    mylogger = logging.getLogger("my")
    mylogger.setLevel(logging.INFO)
    stream_hander = logging.StreamHandler()
    mylogger.addHandler(stream_hander)
    mylogger.info("server start!!!")

    # logging.error("something wrong")