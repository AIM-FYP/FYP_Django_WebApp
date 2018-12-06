
var words = [
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

];


/* Sentitment Summary WordCloud */

//Simple animated example of d3-cloud - https://github.com/jasondavies/d3-cloud
//Based on https://github.com/jasondavies/d3-cloud/blob/master/examples/simple.html

// Encapsulate the word cloud functionality
function wordCloud(selector) {



    var wordScale = d3.scaleLinear().range([7,43]);
    wordScale
        .domain([d3.min(sentiment_summary_word_Data, function(d){return d._size; }),
                 d3.max(sentiment_summary_word_Data, function(d){return d._size; })

    ]);

    var fill = d3.scaleOrdinal(d3.schemeCategory20);
    var width=300;
    var height=300;
    //Construct the word cloud's SVG element
    var svg = d3.select(selector).append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")//;
        .attr("transform", "translate("+(width/2)+","+(height/2)+")");


    //Draw the word cloud
    function draw(sentiment_summary_word_Data) {
        var cloud = svg.selectAll("g text")
                        .data(sentiment_summary_word_Data, function(d) { return d.text; })

        //Entering words
        cloud.enter()
            .append("text")
            .style("font-family", "Impact")
            .style("fill", function(d, i) { return fill(i); })
            .attr("text-anchor", "middle")
            .attr('font-size', 0)//function(d) { return wordScale(d._size) + "px"; }
            .text(function(d) { return d.text; });

        //Entering and existing words
        cloud
            .transition()
                .duration(600)
                .style("font-size", function(d) { return wordScale(d._size) + "px"; })
                .attr("transform", function(d) {
                    return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
                })
                .style("fill-opacity", 1)
                .text(function(d) { return d.text; });

        //Exiting words
        cloud.exit()
            .transition()
                .duration(200)
                .style('fill-opacity', 1e-6)
                .attr('font-size', 1)
                .remove();
    }


    //Use the module pattern to encapsulate the visualisation code. We'll
    // expose only the parts that need to be public.
    return {

        //Recompute the word cloud for a new set of words. This method will
        // asycnhronously call draw when the layout has been computed.
        //The outside world will need to call this function, so make it part
        // of the wordCloud return value.
        update: function(sentiment_summary_word_Data) {
            d3.layout.cloud().size([width, height])
                .words(sentiment_summary_word_Data)
                .padding(0)
                .rotate(function() { return 0/*~~(Math.random() * 2) * 90*/; })
                .font("Impact")
                .fontSize(function(d) { return wordScale(d._size); })
                .on("end", draw)
                .start();
        }
    }

}

//Some sample data - http://en.wikiquote.org/wiki/Opening_lines

//Prepare one of the sample sentences by removing punctuation,
// creating an array of words and computing a random size attribute.
 /*
function getWords() {
    return words;
}*/

//This method tells the word cloud to redraw with a new set of words.
//In reality the new words would probably come from a server request,
// user input or some other source.
function showNewWords(cloudObj) {
    cloudObj.update(sentiment_summary_word_Data/*getWords()*/);
    /* words updater function - aimancomment */
    setTimeout(function() { showNewWords(cloudObj)}, 2000);
}

//Create a new instance of the word cloud visualisation.
var myWordCloud = wordCloud('#wordcloud1');

//Start cycling through the demo data
showNewWords(myWordCloud);

