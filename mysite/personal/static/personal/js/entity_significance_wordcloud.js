

/* Entity Significance WordCloud */
/* 55 words! */



console.log("Data Rcvd : ",entity_significance_wordcloud_Data);
//12,45 before
console.log("Applying (7,43)");
var wordScale = d3.scaleLinear().range([7,43]);
    wordScale
        .domain([d3.min(entity_significance_wordcloud_Data, function(d){return d._size; }),
                 d3.max(entity_significance_wordcloud_Data, function(d){return d._size; })

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

        })*/entity_significance_wordcloud_Data)

    .padding(0)
    .rotate(function(){return 0})
    .font("Impact")
    .fontSize(function(d){return wordScale(d._size);})
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
            .style("font-size",function(d){ return wordScale(d._size);})
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

