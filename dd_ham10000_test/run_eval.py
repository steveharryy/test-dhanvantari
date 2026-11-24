import sys, types, runpy
import tensorflow as tf
m = types.ModuleType('ai_edge_litert'); i = types.ModuleType('ai_edge_litert.interpreter')
i.Interpreter = tf.lite.Interpreter; m.interpreter = i
sys.modules['ai_edge_litert'] = m; sys.modules['ai_edge_litert.interpreter'] = i
sys.argv = ['evaluate_dd.py'] + sys.argv[1:]
runpy.run_path('/Users/abhaypandit/Downloads/untitled folder/evaluate_dd.py', run_name='__main__')
