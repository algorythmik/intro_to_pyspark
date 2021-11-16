#  Guide
## Import statements
Import pyspark sql functions, windows, types narrowly and consistently.
### Good
```
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pysparl.sql import windows as w
```
### Bad
```
from pyspark.sql import *
from pyspark.sql.functions import *
```
