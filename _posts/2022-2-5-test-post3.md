---
layout: post
title: Blog Post 3
---


## Web Scraping


### 1. First Part
My favourite TV in IMDB is an anime, ***Sword Art Online***. So, in this blog, I am going to scrape the actors information and show the result here.
Here is a quick view of the start [URL](https://www.imdb.com/title/tt2250192/).

And I am highly recommend this anime, it's extremely interesting.

![sao1.jpg]({{ site.baseurl }}/images/sao1.jpg)

<br />

### 2. Second Part

In this blog post, I’m going to make a super cool web scraper. Here’s a [link](https://github.com/panxinming/Web-Scraping) to my project repository. Here’s how we set up the project
<br />

#### (a)
First, I write a class to scrape. In this class there are three functions in total. First is parse().

This function has two parts:
- Find the next link, which represents the details of our Movie.
- Once we get into next link, then we are supposed to use the next function to get the detail information of the actors.

```python
def parse(self, response):
    '''The function has two variables one is self and another is response.
    The purpose of this function is to go to the next page we want.
    '''
    # strat from original URL and go to the next.
    url = response.url + "fullcredits/"
    # use request to go to next link and apply next function.
    yield scrapy.Request(url, callback = self.parse_full_credits)

```

#### (b)
This method works by another function parse_full_credits(), so I will introduce this funtion next.

This function has three parts:
- First, we get the next link so that we can go into actor's page.
- Then, once we get into the next page, we need to use the next function to scrapy the information we need.

```python
def parse_full_credits(self, response):
    '''The function has two variables one is self and another is response.
    The purpose of this function is scrap all the actor names, and go to their
    own pages.
    '''
    # this is the hint given by professor, we can get the next link.
    actor = [a.attrib["href"] for a in response.css("td.primary_photo a")]
    # because there are many actors, so we would like to go to many next pages.
    for next in actor:
    # write a URL per itration. 
    url = "https://www.imdb.com" + next
    # then, went to the next link and apply next funtion.
    yield scrapy.Request(url, callback = self.parse_actor_page)

```

#### (c)
This method works by another function parse_actor_page(), so I will introduce this function.

This function has three parts:
- First, we extract the actor's name.
- Then, we are going to extract the movie the actor has worked before.
- Last step, we are going to extract all we need and give them a name. After we done, we are going to put everything we just extract into CSV file.


```python
def parse_actor_page(self, response):
    '''The function has two variables one is self and another is response.
    The purpose of this function is scrap the actor's name and TV or Moive he 
    has worked before.
    '''
    # extract actor's name
    actor_name = response.css("span.itemprop::text").get()
    # get TV or Movie shows which the actor has worked before.
    boxes = response.css("div#content-2-wide.redesign")
    TV_Movie = boxes.css("div.filmo-row b a::text").getall()
    TV_Movie = ",".join(TV_Movie)
    # use for loop, so what we can get one row which contains the actor name and one TV or Movie show.
    for i in TV_Movie.split(","):
        yield{
            "actor":actor_name,
            "Movie_or_TV_name":i
             }

```

<br />

#### (d)
These three funtions build the spider class. Next, I will show you how the class looks like.


```python
import scrapy
import random

class ImdbSpider(scrapy.Spider):
    name = 'imdb_spider'
    start_urls = ['https://www.imdb.com/title/tt2250192/']


    def parse(self, response):
        url = response.url + "fullcredits/"
        yield scrapy.Request(url, callback = self.parse_full_credits)


    def parse_full_credits(self, response):
        actor = [a.attrib["href"] for a in response.css("td.primary_photo a")]
        for next in actor: 
            url = "https://www.imdb.com" + next
            yield scrapy.Request(url, callback = self.parse_actor_page)


    def parse_actor_page(self, response):
            actor_name = response.css("span.itemprop::text").get()
            boxes = response.css("div#content-2-wide.redesign")
            TV_Movie = boxes.css("div.filmo-row b a::text").getall()
            TV_Movie = ",".join(TV_Movie)
            for i in TV_Movie.split(","):
                yield{
                    "actor":actor_name,
                    "Movie_or_TV_name":i
                }
```

<br />
<br />

### 3. Third Part

Because I am happy with the operation of my spider, so I compute a sorted list with the top movies and TV shows that share actors with my favorite movie or TV show.


First, I run this into my terminal, then I can get a CSV file.
```
scrapy crawl imdb_spider -o results.csv
```
<br />
Then, I plan to create a table.

This part is about data cleaning step. Because the information we extract is not sortable. So, I use some code so sort the data, so that we can see which movie has the most shared actors.
```python
import pandas
data = pandas.read_csv("results.csv")
new_data = data.groupby("Movie_or_TV_name").size().reset_index(name='number of shared actors')
df = new_data.sort_values(by=['number of shared actors'], ascending=False)
df.index = list(range(0,10573))
df
```

{% include table.html %}


From this table, we can see that the Anime, Sword Art Online, has the most shared actors.


![sao.jpg]({{ site.baseurl }}/images/sao.jpg)


<br />

<br />

Thank you!