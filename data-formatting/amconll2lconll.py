import jnius_config
jnius_config.set_classpath('.', './amr-labels-helper-all-1.0.jar')

from jnius import autoclass

ArrayList = autoclass('java.util.ArrayList')
# Codec = autoclass("de.up.ling.irtg.algebra.graph.SGraph")

ExtractLabels = autoclass("de.saar.coli.amrtagging.scripts.ExtractLabels")
#el = ExtractLabels()

#print(el.labelsFromGraphString("(n3<root> / --LEX--  :name-of (c2 / county  :location-of (l2<mod>)))"))

Stack = autoclass('java.util.Stack')
stack = Stack()
stack.push('hello')
stack.push('world')
print(stack.pop()) # --> 'world' print stack.pop() # --> 'world'
print(stack.pop()) # --> 'world' print stack.pop() # --> 'hello'