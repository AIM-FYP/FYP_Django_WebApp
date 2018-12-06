
/* Entity Significance Bar Chart */
/*var Data = [
    {
        "entities":['Corruption','Kulsoom'],
        "sentiment":'neg'
    },
    {
        "entities":['leadership','Dignity'],
        "sentiment":'pos'
    },


    {
        "entities":['leadership','replicate'],
        "sentiment":'pos'
    },
    {
        "entities":['Corruption','judiciary'],
        "sentiment":'neg'
    },
    {
        "entities":['leadership','China'],
        "sentiment":'pos'
    },
    {
        "entities":['love','political'],
        "sentiment":'pos'
    },
    {
        "entities":['Corruption','Afghanistan'],
        "sentiment":'neg'
    },
    {
        "entities":['Corruption','NUCES'],
        "sentiment":'neu'
    },
    {
        "entities":['Corruption','Dignity'],
        "sentiment":'pos'
    },
    {
        "entities":['Corruption','hatred'],
        "sentiment":'neg'
    },
    {
        "entities":['leadership','China','peace'],
        "sentiment":'pos'
    },
    {
        "entities":['Pakistani','children'],
        "sentiment":'pos'
    },
    {
        "entities":['Corruption','interfere'],
        "sentiment":'neg'
    },
    {
        "entities":['LAW','died'],
        "sentiment":'pos'
    },
    {
        "entities":['leadership','peace'],
        "sentiment":'pos'
    },
    {
        "entities":['Rights','China'],
        "sentiment":'pos'
    },
    {
        "entities":['Corruption','power'],
        "sentiment":'neg'
    },
    {
        "entities":['marathon','Action'],
        "sentiment":'pos'
    },
    {
        "entities":['leadership','Human'],
        "sentiment":'pos'
    },
    {
        "entities":['metropolism','state'],
        "sentiment":'neg'
    },
    {
        "entities":['leadership','breaking'],
        "sentiment":'pos'
    },
    {
        "entities":['leadership','peace'],
        "sentiment":'pos'
    },
    {
        "entities":['depression','hatred'],
        "sentiment":'neg'
    },
    {
        "entities":['Corruption','India'],
        "sentiment":'neg'
    },
    {
        "entities":['basis','rape'],
        "sentiment":'neg'
    },
    {
        "entities":['leadership','died'],
        "sentiment":'neu'
    },
    {
        "entities":['Corruption','conspiracy'],
        "sentiment":'neg'
    },
    {
        "entities":['leadership','quran'],
        "sentiment":'pos'
    },
    {
        "entities":['marathon','basis'],
        "sentiment":'neu'
    },
    {
        "entities":['Corruption','conspiracy'],
        "sentiment":'neg'
    },
    {
        "entities":['basis','fork'],
        "sentiment":'neu'
    },
    {
        "entities":['research','action'],
        "sentiment":'neu'
    },
    {
        "entities":['research','retro'],
        "sentiment":'neu'
    }
];*/

function getStats(word){
    var posCount=0;
    var negCount=0;
    var neuCount=0;

    if (word=='#none'){

        entity_significance_bar_Data.forEach(function(d) {
            if (d.sentiment=='pos'){
                posCount+=1;
            }
            else if (d.sentiment=='neg'){
                negCount+=1;
            }
            else{
                neuCount+=1;
            }
        });

    }
    else{

        entity_significance_bar_Data.forEach(function(d) {
            if (d.entities.includes(word)){
                if (d.sentiment=='pos'){
                posCount+=1;
                }
                else if (d.sentiment=='neg'){
                    negCount+=1;
                }
                else{
                    neuCount+=1;
                }
            }

        });

    }



    return [posCount, negCount, neuCount];

}


var initialState = getStats('#none');

var dynamicBarChart = c3.generate({
    bindto: '#barchart1',
    data: {
        columns: [
            ['pos', initialState[0]],
            ['neg', initialState[1]],
            ['neu', initialState[2]]
        ],
        type: 'bar'
    },
    color: {
        pattern: ['#38ed62', '#e35e57', '#4a69e6']

    },
    bar: {
        width: {
           //ratio: 0.5 // this makes bar width 50% of length between ticks
            // or
            //width: 100 // this makes bar width 100px
        }

    },
    size: {
        width: 300,
        height: 300
    }
});
