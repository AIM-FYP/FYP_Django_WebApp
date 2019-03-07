// %m/%d/%Y %H:%M           "1/1/2013 02:00" 

// %Y-%m-%d %H:%M          "2018-12-05 15:00"
/*
            var event_chart_Data=[
            {
                "time": "2018-12-05 15:00",
                "pos" : 12,
                "neg" : 30,
                "neu" : 33,
                "event": "none",
                "event-title": "none",
                "event-date": "none",
                "event-source": "none",
                "event-url": "none",
                "event-imgsrc": "none"
              
            },
            {
                "time": "2018-12-05 16:00",
                "pos" : 68,
                "neg" : 20,
                "neu" : 22,
                "event": "none",
                "event-title": "none",
                "event-date": "none",
                "event-source": "none",
                "event-url": "none",
                "event-imgsrc": "none"
               
            },
            {
                "time": "2018-12-05 17:00",
                "pos" : 68,
                "neg" : 35,
                "neu" : 12,
                "event": "none",
                "event-title": "none",
                "event-date": "none",
                "event-source": "none",
                "event-url": "none",
                "event-imgsrc": "none"
              
            },
            {
                "time": "2018-12-05 18:00",
                "pos" : 33,
                "neg" : 29,
                "neu" : 88,
                "event": "none",
                "event-title": "none",
                "event-date": "none",
                "event-source": "none",
                "event-url": "none",
                "event-imgsrc": "none"
               
            },
            {
                "time": "2018-12-05 19:00",
                "pos" : 21,
                "neg" : 45,
                "neu" : 8,
                "event": "none",
                "event-title": "none",
                "event-date": "none",
                "event-source": "none",
                "event-url": "none",
                "event-imgsrc": "none"
              
            },
            {
                "time": "2018-12-05 20:00",
                "pos" : 34,
                "neg" : 67,
                "neu" : 9,
                "event": "none",
                "event-title": "none",
                "event-date": "none",
                "event-source": "none",
                "event-url": "none",
                "event-imgsrc": "none"
               
            },
            {
                "time": "2018-12-05 21:00",
                "pos" : 43,
                "neg" : 87,
                "neu" : 15,
                "event": "event",
                "event-title": "US wants to know if Pakistan used F-16 jets to down Indian fighter plane",
                "event-date": "03-Mar-2019",
                "event-source": "Business Today",
                "event-url": "https://www.businesstoday.in/current/world/us-wants-to-know-if-pakistan-used-f-16-jets-to-down-indian-fighter-plane/story/324200.html",
                "event-imgsrc": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRUftlu5wip923mYQzHo-ndQgoXApjhfeTrv4dif5yxeNEwBi9YJz5OKkQrehy1rmp2Riip1TE"
              
            },
            {
                "time": "2018-12-05 22:00",
                "pos" : 33,
                "neg" : 23,
                "neu" : 28,
                "event": "none",
                "event-title": "none",
                "event-date": "none",
                "event-source": "none",
                "event-url": "none",
                "event-imgsrc": "none"
               
            }
            ];
*/

    console.log("event event event data here : ",event_chart_Data);
    
    var svg = d3.select("svg#events"),
        margin = {top: 20, right: 20, bottom: 110, left: 40},
        margin2 = {top: 430, right: 20, bottom: 30, left: 40},
        width = +svg.attr("width") - margin.left - margin.right,
        height = +svg.attr("height") - margin.top - margin.bottom,
        height2 = +svg.attr("height") - margin2.top - margin2.bottom;

    var parseDate = d3.timeParse("%Y-%m-%d %H:%M");

    var x = d3.scaleTime().range([0, width]),
        x2 = d3.scaleTime().range([0, width]),
        y = d3.scaleLinear().range([height, 0]),
        y2 = d3.scaleLinear().range([height2, 0]);

    var xAxis = d3.axisBottom(x),
        xAxis2 = d3.axisBottom(x2),
        yAxis = d3.axisLeft(y);

    var brush = d3.brushX()
        .extent([[0, 0], [width, height2]])
        .on("brush end", brushed);

    var zoom = d3.zoom()
        .scaleExtent([1, Infinity])
        .translateExtent([[0, 0], [width, height]])
        .extent([[0, 0], [width, height]])
        .on("zoom", zoomed);

        var line = d3.line()
            .x(function (d) { return x(d.time); })
            .y(function (d) { return y(d.pos); });
    
        var line2 = d3.line()
            .x(function (d) { return x(d.time); })
            .y(function (d) { return y(d.neg); });
    
        var line3 = d3.line()
            .x(function (d) { return x(d.time); })
            .y(function (d) { return y(d.neu); });
        
        var context_line = d3.line()
            .x(function (d) { return x2(d.time); })
            .y(function (d) { return y2(d.pos); });
        var context_line2 = d3.line()
            .x(function (d) { return x2(d.time); })
            .y(function (d) { return y2(d.neg); });
        var context_line3 = d3.line()
            .x(function (d) { return x2(d.time); })
            .y(function (d) { return y2(d.neu); });

    
        var clip = svg.append("defs").append("svg:clipPath")
            .attr("id", "clip")
            .append("svg:rect")
            .attr("width", width)
            .attr("height", height)
            .attr("x", 0)
            .attr("y", 0);

   
        var Line_chart = svg.append("g")
            .attr("class", "focus")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
            .attr("clip-path", "url(#clip)");
    

        var focus = svg.append("g")
            .attr("class", "focus")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    
    var context = svg.append("g")
        .attr("class", "context")
        .attr("transform", "translate(" + margin2.left + "," + margin2.top + ")");


    
    
      x.domain(d3.extent(event_chart_Data, function(d) { return type(d).time; }));
      var ymax = d3.max([d3.max(event_chart_Data, function (d) { return d.pos; }),
                        d3.max(event_chart_Data, function (d) { return d.neg; }),
                        d3.max(event_chart_Data, function (d) { return d.neu; })]);
      y.domain([0, ymax+10]);//d3.max(event_chart_Data, function (d) { return d.pos; })]);
      x2.domain(x.domain());
      y2.domain(y.domain());


        focus.append("g")
            .attr("class", "axis axis--x")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis);

        focus.append("g")
            .attr("class", "axis axis--y")
            .call(yAxis);

    
         
        
	  

    
        Line_chart.append("path")
            .datum(event_chart_Data)
            .attr("class", "line")
            .attr("d", line)
            .attr("id","line-pos")
            .style("stroke", '#38ed62' );
    
        Line_chart.append("path")
            .datum(event_chart_Data)
            .attr("class", "line")
            .attr("d", line2)
            .attr("id","line-neg")
            .style("stroke", '#e35e57' );
    
       
    
        Line_chart.append("path")
            .datum(event_chart_Data)
            .attr("class", "line")
            .attr("d", line3)
            .attr("id","line-neu")
            .style("stroke", '#4a69e6' );
    
       
        context.append("path")
            .datum(event_chart_Data)
            .attr("class", "line")
            .attr("d", context_line)
            .style("stroke", '#38ed62' );
    
        context.append("path")
            .datum(event_chart_Data)
            .attr("class", "line")
            .attr("d", context_line2)
            .style("stroke", '#e35e57' );
    
        context.append("path")
            .datum(event_chart_Data)
            .attr("class", "line")
            .attr("d", context_line3)
            .style("stroke", '#4a69e6' );

        var color_code = ['#38ed62' , '#e35e57' , '#4a69e6'];
        var yaxis_param = ['pos', 'neg' , 'neu'];

   
    // Add legend   
	var legend = svg.append("g")
	  .attr("class", "legend")
	  .attr("height", 100)
	  .attr("width", 100)
      .attr('transform', 'translate(80,10)');   
          
    
    legend.selectAll('rect')
      .data(yaxis_param)
      .enter()
      .append("rect")
	  .attr("x", width - 65)
      .attr("y", function(d, i){ return i *  20;})
	  .attr("width", 10)
	  .attr("height", 10)
	  .style("fill", function(d,i) { 
        var color = color_code[i];
        return color;
      });
    
     legend.selectAll('text')
      .data(yaxis_param)
      .enter()
      .append("text")
	  .attr("x", width - 52)
      .attr("y", function(d, i){ return i *  20 + 9;})
	  .text(function(d,i) {
        var text = d;
        return text;
      });
    
  
    
    //legend end
    
    //zoom
    Line_chart.append("rect")
      .attr("class", "zoom")
      .attr("width", width)
      .attr("height", height)
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
      .call(zoom);

    
    var tip = d3.tip()
      .attr('class', 'd3-tip')
      .offset([-10, 0])
      .html(function(d) {
          if (d.event=="none")
              return ""
          return "<table border='0'><tr>"+
                "<td align='left'>  <img style='vertical-align:bottom;' width='100%' height='100%' src='"+d["event-imgsrc"]+"'> </td>"+
                "<td align='left' style='vertical-align: top;'> <strong>"+d["event-title"]+"</strong> <br><br><br><br>"+
                " <span style='color:grey'>" + d["event-source"] + "</span> <br>"+
                " <span style='font-size: 10px;color:grey'>" + d["event-date"] + "</span> </td></tr></table>";
      });
    
    svg.call(tip);
    
    var circlePoint1 = Line_chart.selectAll(".circle#circle1")
        .data(event_chart_Data).enter()
          .append("circle")
          .attr("class", "circle")
          .attr("r", 3)
          .attr("cx", function(d) {
            return x(d.time);
          })
          .attr("cy", function(d) {
            return y(d.pos);
          })
          .style("fill", '#38ed62' )
          // event tool tip
          .on('mouseover', function(d,i){
              if (d.event!="none")
                  tip.show(d,i);
          })
          .on('mouseout', function(d){
              if (d.event!="none")
                  tip.hide(d);
          });
    
    
    var circlePoint2 =Line_chart.selectAll(".circle#circle2")
        .data(event_chart_Data).enter()
          .append("circle")
          .attr("class", "circle")
          .attr("r", 3)
          .attr("cx", function(d) {
            return x(d.time);
          })
          .attr("cy", function(d) {
            return y(d.neg);
          })
          .style("fill", '#e35e57' )
          // event tool tip
          .on('mouseover', function(d,i){
              if (d.event!="none")
                  tip.show(d,i);
          })
          .on('mouseout', function(d){
              if (d.event!="none")
                  tip.hide(d);
          });
    

    var circlePoint3 =Line_chart.selectAll(".circle#circle3")
        .data(event_chart_Data).enter()
          .append("circle")
          .attr("class", "circle")
          .attr("r", 3)
          .attr("cx", function(d) {
            return x(d.time);
          })
          .attr("cy", function(d) {
            return y(d.neu);
          })
          .style("fill", '#4a69e6' )
          // event tool tip
          .on('mouseover', function(d,i){
              if (d.event!="none")
                  tip.show(d,i);
          })
          .on('mouseout', function(d){
              if (d.event!="none")
                  tip.hide(d);
          });
    

    
    
    
      context.append("g")
          .attr("class", "axis axis--x")
          .attr("transform", "translate(0," + height2 + ")")
          .call(xAxis2);

      context.append("g")
          .attr("class", "brush").call(brush)
          .call(brush.move, x.range());


    
 

    function brushed() {
      if (d3.event.sourceEvent && d3.event.sourceEvent.type === "zoom") return; // ignore brush-by-zoom
      var s = d3.event.selection || x2.range();
      x.domain(s.map(x2.invert, x2));
        
      
      Line_chart.select("path#line-pos")
          .attr("d",line);    
      Line_chart.select("path#line-neg")
          .attr("d",line2);  
      Line_chart.select("path#line-neu")
          .attr("d",line3);
      
      circlePoint1
      .attr("cx", function (d) { return x(d.time); })
      .attr("cy", function (d) { return y(d.pos); });
        
      circlePoint2
      .attr("cx", function (d) { return x(d.time); })
      .attr("cy", function (d) { return y(d.neg); });
        
      circlePoint3
      .attr("cx", function (d) { return x(d.time); })
      .attr("cy", function (d) { return y(d.neu); });
        
        
        
      focus.select(".axis--x").call(xAxis);
      svg.select(".zoom").call(zoom.transform, d3.zoomIdentity
          .scale(width / (s[1] - s[0]))
          .translate(-s[0], 0));
    }

    function zoomed() {
      if (d3.event.sourceEvent && d3.event.sourceEvent.type === "brush") return; // ignore zoom-by-brush
      var t = d3.event.transform;
      x.domain(t.rescaleX(x2).domain());
        
      
      Line_chart.select("path#line-pos")
          .attr("d",line);    
      Line_chart.select("path#line-neg")
          .attr("d",line2);  
      Line_chart.select("path#line-neu")
          .attr("d",line3);  
        
       
      circlePoint1
      .attr("cx", function (d) { return x(d.time); })
      .attr("cy", function (d) { return y(d.pos); });
        
      circlePoint2
      .attr("cx", function (d) { return x(d.time); })
      .attr("cy", function (d) { return y(d.neg); });
        
      circlePoint3
      .attr("cx", function (d) { return x(d.time); })
      .attr("cy", function (d) { return y(d.neu); });
    
            
        
        
      focus.select(".axis--x").call(xAxis);
      context.select(".brush").call(brush.move, x.range().map(t.invertX, t));
        
    }
    
    function type(d) {
      d.time = parseDate(d.time);
      d.pos = +d.pos;
      d.neg = +d.neg;
      d.neu = +d.neu;
      return d;
    }