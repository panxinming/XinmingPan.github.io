---
layout: post
title: Plotly Example
---

Fortunately, it's pretty easy to embed interactive HTML figures produced via Plotly on your blog. Just use  `plotly.io.write_html()` to save your figure. Then, copy the resulting HTML file to the `_includes` directory of your blog. Finally, place the code  

```
{{ "{% include example_fig.html " }}%}
```

at the location of your blog at which you would like the figure to appear. Here's an example: 


```python
import plotly.graph_objects as go
from plotly.io import write_html

# from the plotly tutorial
fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
write_html(fig, "example_fig.html")

# manually copy example_fig.html to _includes directory of blog
```

{% include example_fig.html %}