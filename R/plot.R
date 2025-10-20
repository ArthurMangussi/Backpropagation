library(DiagrammeR)
library(dotty)

plot_graph <- function(value, ...) {
  graph <- create_graph() %>% 
    add_global_graph_attrs("rankdir", "LR", "graph") %>% 
    add_global_graph_attrs("ranksep", "equally", "graph") %>% 
    add_global_graph_attrs("layout", "dot", "graph")
    
    .[graph, ..] <- plot_graph_impl(graph, value)
    graph <- graph %>% 
      select_nodes(conditions = type == "data") %>% 
      set_node_attrs_ws(node_attr = shape, value = "record") %>% 
      set_node_attr_to_display("d") %>% 
      clear_selection() %>% 
      set_node_attrs(fontsize, 12) %>% 
      set_node_attrs(node_attr = "fixedsize", values = FALSE) %>%
      set_graph_directed()
      
    render_graph(graph, ...)
}

plot_graph_impl <- function(graph, v, parent = NULL, visited = c()) {
  if (v$id %in% visited) { 
    return(list(graph, visited))
  }
  visited <- c(visited, v$id)

# Add itself to the graph
  display <- if (length(v$name) && v$name != "") {
    glue::glue("{v$name} | data: {format(v$data)}")
  } else {
    glue::glue("value: {format(v$data)}")
  }

  if (!is.null(v$grad)) {
    display <- glue::glue("{display} | grad: {format(v$grad)}")
  }
  
  graph <- graph %>% 
    add_node(
      label = v$id,
      node_data = node_data(d = display),
      type = "data"
    )

# Add the op to the graph
  if (v$op != "") {
    graph <- graph %>% 
      add_node(
        label = paste0(v$id, "_op"),
        node_data = node_data(d = v$op),
        type = "op"
      ) %>%
      add_edge(from = paste0(v$id, "_op"), to = v$id)
  }

# Add edges to parent
  if (!is.null(parent)) {
    graph <- graph %>% add_edge(
      from = as.character(v$id),
      to = paste0(parent$id, "_op")
    )
  }

# Add children
  for (child in v$children) { 
    .[graph, visited] <- plot_graph_impl(graph, child, parent = v, visited = visited)
    }

  list(graph, visited)
}

































