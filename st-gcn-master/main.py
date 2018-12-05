#!/usr/bin/env python
import argparse
import sys
import time
# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo'] = import_class('processor.demo.Demo')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])
    start_time=time.time()
    try:
        p.start()
    except Exception as e:
        print(e)
    end_time=time.time()
    print("application total time: %s s" % str(end_time-start_time))
    

