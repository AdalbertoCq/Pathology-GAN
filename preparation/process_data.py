#!/usr/bin/python

import preparation.preprocessor as pre

with pre.Preprocessor() as preprocessor:
    preprocessor.run()