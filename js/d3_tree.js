'use strict';

var width = 960,
    height = 500

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height);

var tree = d3.tree().size([height, width]);

var nodes = d3.hierarchy(iris);