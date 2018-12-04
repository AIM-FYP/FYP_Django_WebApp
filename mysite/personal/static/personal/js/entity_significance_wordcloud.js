

/* Entity Significance WordCloud */

var Data = [
    {
        'text':'scandals',
        'size':37
    },
    {
        'text':'conspiracy',
        'size':22
    },
    {
        'text':'marathon',
        'size':12
    },
    {
        'text':'depression',
        'size':18
    },
    {
        'text':'children',
        'size':13
    },
    {
        'text':'women',
        'size':9
    },
    {
        'text':'LAW',
        'size':23
    },
    {
        'text':'man',
        'size':5
    },
    {
        'text':'power',
        'size':8
    },
    {
        'text':'breaking',
        'size':11
    },
    {
        'text':'NUCES',
        'size':12
    },
    {
        'text':'Dignity',
        'size':45
    },
    {
        'text':'Corruption',
        'size':72
    },
    {
        'text':'Pakistani',
        'size':58
    },
    {
        'text':'Ethics',
        'size':27
    },
    {
        'text':'replicate',
        'size':22
    },
    {
        'text':'judiciary',
        'size':14
    },
    {
        'text':'Government',
        'size':31
    },
    {
        'text':'research',
        'size':9
    },
    {
        'text':'Action',
        'size':20
    },
    {
        'text':'leadership',
        'size':40
    }
    ,
    {
        'text':'racism',
        'size':22
    }
    ,
    {
        'text':'fork',
        'size':23
    }
    ,
    {
        'text':'Army',
        'size':25
    }
    ,
    {
        'text':'interfere',
        'size':28
    }
    ,
    {
        'text':'hate',
        'size':30
    }
    ,
    {
        'text':'love',
        'size':27
    }
    ,
    {
        'text':'rape',
        'size':25
    }
    ,
    {
        'text':'minority',
        'size':36
    }
    ,
    {
        'text':'Human',
        'size':23
    }
    ,
    {
        'text':'Rights',
        'size':34
    }
    ,
    {
        'text':'Honor',
        'size':18
    }
    ,
    {
        'text':'role',
        'size':32
    }
    ,
    {
        'text':'metropolism',
        'size':12
    }
    ,
    {
        'text':'died',
        'size':33
    }
    ,
    {
        'text':'India',
        'size':14
    }
    ,
    {
        'text':'China',
        'size':32
    }
    ,
    {
        'text':'Afghanistan',
        'size':26
    }
    ,
    {
        'text':'Nawaz',
        'size':30
    }
    ,
    {
        'text':'Kulsoom',
        'size':33
    }
    ,
    {
        'text':'state',
        'size':31
    }
    ,
    {
        'text':'radicate',
        'size':23
    }
    ,
    {
        'text':'quran',
        'size':20
    }
    ,
    {
        'text':'lies',
        'size':24
    }
    ,
    {
        'text':'hatred',
        'size':21
    }
    ,
    {
        'text':'Anger',
        'size':33
    }
    ,
    {
        'text':'emotions',
        'size':21
    }
    ,
    {
        'text':'peace',
        'size':30
    }
    ,
    {
        'text':'vision',
        'size':26
    }
    ,
    {
        'text':'basis',
        'size':18
    }
    ,
    {
        'text':'Political',
        'size':25
    }
    ,
    {
        'text':'tweak',
        'size':33
    }
    ,
    {
        'text':'retro',
        'size':27
    }

];

var wordScale = d3.scaleLinear().range([6,70]);
    wordScale
        .domain([d3.min(Data, function(d){return d.size; }),
                 d3.max(Data, function(d){return d.size; })

    ]);

var fill = d3.scaleOrdinal(d3.schemeCategory20);
var width=690;
var height=300;
var topMargin=7;
d3.layout.cloud().size([width,height])
    .words(/*[
      "hello","world","tanned","hmm","maira","fyp","world","to","me"  
        ].map(function(d){
            return {text:d, size: 10+Math.random()*90};

        })*/Data)
    .padding(0)
    .rotate(function(){return 0})
    .font("Impact")
    .fontSize(function(d){return wordScale(d.size);})
    .on("end", draw)
    .start();

function draw(words){
    d3.select("#wordcloud2").append("svg")
            .attr("width", width)
            .attr("height", height)
        .append("g")
            .attr("transform", "translate("+(width/2)+","+(height/2+topMargin)+")")
        .selectAll("text")
            .data(words)
        .enter().append("text")
            .style("font-size",function(d){ return wordScale(d.size);})
            .style("font-family","Impact")
            .style("fill", function(d,i){ return fill(i);})
            .attr("text-anchor", "middle")
            .attr("transform", function(d){
                return "translate("+[d.x,d.y]+")rotate("+d.rotate+")";  
            })
            .text(function(d){  return d.text;  })
            .on("click", animate)   
            .on("mouseover", greyAll)

            .on("mouseout",  colorAll );
}

function animate(d){

}

function greyAll(d){

    /* Grey Out Each Word */
    d3.select('#wordcloud2')
    .select("svg")
    .select("g")
    .selectAll("text")
        .style('fill', '#e3e1df'); 

    /* Add custom color to mouse over word */
    d3.select(this)
            .style('fill', 'darkOrange'); 

    /* Update bar chart as selected color */
    var selection = d.text;
    var update = getStats(selection);

    dynamicBarChart.load({
      columns: [
        ['pos', update[0] ],
        ['neg', update[1] ],
        ['neu', update[2] ]
      ]
    });



}

function colorAll(d){

    /* Color all words back */
    d3.select('#wordcloud2')
    .select("svg")
    .select("g")
    .selectAll("text")
        .style('fill', function(d,i){
         return fill(i);
        });  

    /* Revert bar chart to none selection */
    var update = getStats('#none');

    dynamicBarChart.load({
      columns: [
        ['pos', update[0] ],
        ['neg', update[1] ],
        ['neu', update[2] ]
      ]
    });
}

