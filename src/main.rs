extern crate ordered_float;

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::{Cell, RefCell};
use std::ops::{Add, Mul, Sub, Div, Neg};
use ordered_float::OrderedFloat;
use std::cmp;


#[derive(PartialEq, Eq)]
pub enum Inline {
    Yes,
    No,
}

fn main() {
    let graph = Graph::new();
    let x = graph.scalar("x");

    println!("Logistic regression:");
    let target = graph.scalar("y");
    let weights = graph.scalar("W");
    let bias = graph.scalar("b");

    let activation = &x * &weights + &bias;
    let output = sigmoid(&activation);

    let loss = binary_crossentropy(&output, &target);
    let w_grad = loss.gradient(&weights);
    let b_grad = loss.gradient(&bias);

    graph.dump_program(&[loss, w_grad, b_grad], Inline::Yes);

    let graph = Graph::new();
    let x = graph.scalar("x");
    println!("");
    println!("Many `cos`:");
    graph.dump_program(&[x.cos().cos().cos().cos().cos().cos().cos().cos().cos().gradient(&x)],
                       Inline::Yes);


}

fn sigmoid(x: &Node) -> Node {
    1.0 / (1.0 + (-x).exp())
}

fn binary_crossentropy(output: &Node, target: &Node) -> Node {
    -(target * output.log() + (1.0 - target) * (1.0 - output).log())
}


#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct NodeId(usize);

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct SymbolId(usize);

#[derive(Default)]
pub struct GraphData {
    node_to_id: RefCell<HashMap<NodeDef, NodeId>>,
    id_to_node: RefCell<HashMap<NodeId, NodeDef>>,
    id_to_symbol: RefCell<HashMap<SymbolId, SymbolDef>>,
    gradients: RefCell<HashMap<(NodeId, NodeId), NodeId>>,
    contains: RefCell<HashMap<(NodeId, NodeId), bool>>,

    next_node_id: Cell<usize>,
    next_symbol_id: Cell<usize>,
}

impl GraphData {
    fn node(&self, mut def: NodeDef) -> NodeId {
        loop {
            let new_def = self.simplify(def);
            if new_def == def {
                break;
            }
            def = new_def;
        }
        let mut node_to_id = self.node_to_id.borrow_mut();
        let mut id_to_node = self.id_to_node.borrow_mut();
        *node_to_id.entry(def).or_insert_with(|| {
            let id = self.next_node_id.get();
            self.next_node_id.set(id + 1);
            let id = NodeId(id);
            id_to_node.insert(id, def);
            id
        })
    }

    fn simplify(&self, def: NodeDef) -> NodeDef {
        match def {
            NodeDef::Unary(op, of) => {
                match self.node_by_id(of) {
                    NodeDef::Constant(value) => NodeDef::Constant(op.eval(value.into()).into()),
                    NodeDef::Unary(Unary::Log, sub) if op == Unary::Exp => self.node_by_id(sub),
                    NodeDef::Unary(Unary::Exp, sub) if op == Unary::Log => self.node_by_id(sub),
                    NodeDef::Unary(Unary::Neg, sub) if op == Unary::Neg => self.node_by_id(sub),
                    NodeDef::Binary(Binary::Sub, lhs, rhs) if op == Unary::Neg => {
                        NodeDef::Binary(Binary::Sub, rhs, lhs)
                    }
                    _ => def,
                }
            }

            NodeDef::Binary(op, lhs, rhs) => {
                let def = if lhs == rhs {
                    match op {
                        Binary::Add => {
                            NodeDef::Binary(Binary::Mul,
                                            lhs,
                                            self.node(NodeDef::Constant(2.0.into())))
                        }
                        Binary::Sub => NodeDef::Constant(0.0.into()),
                        Binary::Div => NodeDef::Constant(1.0.into()),
                        Binary::Mul => {
                            NodeDef::Binary(Binary::Pow,
                                            lhs,
                                            self.node(NodeDef::Constant(2.0.into())))
                        }
                        Binary::Pow => def,
                    }
                } else {
                    def
                };

                match (self.node_by_id(lhs), self.node_by_id(rhs)) {
                    (NodeDef::Constant(lhs_value), NodeDef::Constant(rhs_value)) => {
                        NodeDef::Constant(op.eval(lhs_value.into(), rhs_value.into()).into())
                    }
                    (NodeDef::Unary(op_lhs, sub_lhs), NodeDef::Unary(op_rhs, sub_rhs))
                        if (op == Binary::Mul || op == Binary::Div) && op_lhs == op_rhs &&
                           (op_lhs == Unary::Exp || op_lhs == Unary::Log) => {
                        let sub_op = if op == Binary::Mul {
                            Binary::Add
                        } else {
                            Binary::Sub
                        };

                        NodeDef::Unary(op_lhs, self.node(NodeDef::Binary(sub_op, sub_lhs, sub_rhs)))
                    }
                    (NodeDef::Unary(Unary::Neg, sub_lhs), NodeDef::Unary(Unary::Neg, sub_rhs)) => {
                        match op {
                            Binary::Mul | Binary::Div => NodeDef::Binary(op, sub_lhs, sub_rhs),
                            Binary::Add => {
                                NodeDef::Unary(Unary::Neg,
                                               self.node(NodeDef::Binary(op, sub_lhs, sub_rhs)))
                            }
                            Binary::Sub => NodeDef::Binary(op, sub_rhs, sub_lhs),
                            _ => def,
                        }
                    }
                    (NodeDef::Binary(Binary::Div, lhs1, rhs1),
                     NodeDef::Binary(Binary::Div, lhs2, rhs2)) => {
                        match op {
                            Binary::Div => {
                                NodeDef::Binary(Binary::Div,
                                                self.node(NodeDef::Binary(Binary::Mul, lhs1, rhs2)),
                                                self.node(NodeDef::Binary(Binary::Mul, lhs2, rhs1)))
                            }
                            Binary::Mul => {
                                NodeDef::Binary(Binary::Div,
                                                self.node(NodeDef::Binary(Binary::Mul, lhs1, lhs2)),
                                                self.node(NodeDef::Binary(Binary::Mul, rhs1, rhs2)))
                            }
                            _ => def,
                        }
                    }
                    (NodeDef::Binary(Binary::Div, lhs1, rhs1), _) if op == Binary::Div ||
                                                                     op == Binary::Mul => {
                        match op {
                            Binary::Div => {
                                NodeDef::Binary(Binary::Div,
                                                lhs1,
                                                self.node(NodeDef::Binary(Binary::Mul, rhs1, rhs)))
                            }
                            Binary::Mul => {
                                NodeDef::Binary(Binary::Div,
                                                self.node(NodeDef::Binary(Binary::Mul, lhs1, rhs)),
                                                rhs1)
                            }
                            _ => unreachable!(),
                        }
                    }
                    (_, NodeDef::Binary(Binary::Div, lhs2, rhs2)) if op == Binary::Div ||
                                                                     op == Binary::Mul => {
                        match op {
                            Binary::Div => {
                                NodeDef::Binary(Binary::Div,
                                                self.node(NodeDef::Binary(Binary::Mul, lhs, rhs2)),
                                                lhs2)
                            }
                            Binary::Mul => {
                                NodeDef::Binary(Binary::Div,
                                                self.node(NodeDef::Binary(Binary::Mul, lhs, lhs2)),
                                                rhs2)
                            }
                            _ => unreachable!(),
                        }
                    }
                    (NodeDef::Constant(OrderedFloat(0.0)), rhs_def) => {
                        match op {
                            Binary::Add => rhs_def,
                            Binary::Sub => NodeDef::Unary(Unary::Neg, rhs),
                            Binary::Mul | Binary::Div => NodeDef::Constant(0.0.into()),
                            Binary::Pow => NodeDef::Constant((-1.0).into()),
                        }
                    }
                    (NodeDef::Constant(OrderedFloat(1.0)), rhs_def) if op == Binary::Mul ||
                                                                       op == Binary::Pow => {
                        match op {
                            Binary::Mul => rhs_def,
                            Binary::Pow => NodeDef::Constant(1.0.into()),
                            _ => unreachable!(),
                        }
                    }
                    (NodeDef::Constant(OrderedFloat(-1.0)), _) if op == Binary::Mul ||
                                                                  op == Binary::Div ||
                                                                  op == Binary::Pow => {
                        match op {
                            Binary::Mul => NodeDef::Unary(Unary::Neg, rhs),
                            Binary::Div => NodeDef::Unary(
                                    Unary::Neg,
                                    self.node(
                                        NodeDef::Binary(
                                            Binary::Div,
                                            self.node(NodeDef::Constant(OrderedFloat(1.0))),
                                            rhs))),
                            Binary::Pow => NodeDef::Constant(1.0.into()),
                            _ => unreachable!(),
                        }
                    }
                    (lhs_def, NodeDef::Constant(OrderedFloat(0.0))) => {
                        match op {
                            Binary::Add | Binary::Sub => lhs_def,
                            Binary::Mul => NodeDef::Constant(0.0.into()),
                            Binary::Pow => NodeDef::Constant(1.0.into()),
                            Binary::Div => panic!("Static division by zero!"),
                        }
                    }
                    (lhs_def, NodeDef::Constant(OrderedFloat(1.0))) if op == Binary::Mul ||
                                                                       op == Binary::Div ||
                                                                       op == Binary::Pow => lhs_def,
                    (_, NodeDef::Constant(OrderedFloat(-1.0))) if op == Binary::Mul ||
                                                                  op == Binary::Div ||
                                                                  op == Binary::Pow => {
                        match op {
                            Binary::Mul | Binary::Div => NodeDef::Unary(Unary::Neg, lhs),
                            Binary::Pow => {
                                NodeDef::Binary(Binary::Div,
                                                self.node(NodeDef::Constant(1.0.into())),
                                                lhs)
                            }
                            _ => unreachable!(),
                        }
                    }
                    (NodeDef::Unary(Unary::Neg, sub_lhs), _) if op == Binary::Mul ||
                                                                op == Binary::Div ||
                                                                op == Binary::Add ||
                                                                op == Binary::Sub => {
                        match op {
                            Binary::Mul | Binary::Div => {
                                NodeDef::Unary(Unary::Neg,
                                               self.node(NodeDef::Binary(op, sub_lhs, rhs)))
                            }
                            Binary::Add => NodeDef::Binary(Binary::Sub, rhs, sub_lhs),
                            Binary::Sub => {
                                NodeDef::Unary(Unary::Neg,
                                               self.node(NodeDef::Binary(Binary::Add,
                                                                         sub_lhs,
                                                                         rhs)))
                            }
                            _ => def,
                        }
                    }
                    (_, NodeDef::Unary(Unary::Neg, sub_rhs)) if op == Binary::Mul ||
                                                                op == Binary::Div ||
                                                                op == Binary::Add ||
                                                                op == Binary::Sub => {
                        match op {
                            Binary::Mul | Binary::Div => {
                                NodeDef::Unary(Unary::Neg,
                                               self.node(NodeDef::Binary(op, lhs, sub_rhs)))
                            }
                            Binary::Add => NodeDef::Binary(Binary::Sub, lhs, sub_rhs),
                            Binary::Sub => NodeDef::Binary(Binary::Add, lhs, sub_rhs),
                            _ => unreachable!(),
                        }
                    }
                    (_, _) => def,
                }
            }

            _ => def,
        }
    }

    fn contains(&self, needle: NodeId, haystack: NodeId) -> bool {
        if needle == haystack {
            return true;
        }

        if let Some(&result) = self.contains.borrow().get(&(needle, haystack)) {
            return result;
        }
        let contains = match self.node_by_id(haystack) {
            NodeDef::Constant(..) |
            NodeDef::Symbol(..) => false,
            NodeDef::Unary(_, of_id) => self.contains(needle, of_id),
            NodeDef::Binary(_, lhs_id, rhs_id) => {
                self.contains(needle, lhs_id) || self.contains(needle, rhs_id)
            }
        };
        self.contains.borrow_mut().insert((needle, haystack), contains);
        contains
    }

    fn symbol(&self, def: SymbolDef) -> NodeId {
        let id = self.next_symbol_id.get();
        self.next_symbol_id.set(id + 1);
        let id = SymbolId(id);
        self.id_to_symbol.borrow_mut().insert(id, def);
        self.node(NodeDef::Symbol(id))
    }

    fn node_by_id(&self, id: NodeId) -> NodeDef {
        *self.id_to_node.borrow().get(&id).expect("")
    }

    fn gradient(&self, objective: NodeId, wrt: NodeId) -> NodeId {
        if let Some(&gradient) = self.gradients.borrow().get(&(objective, wrt)) {
            return gradient;
        }
        let gradient = self.gradient_nocache(objective, wrt);
        self.gradients.borrow_mut().insert((objective, wrt), gradient);
        gradient
    }

    fn gradient_nocache(&self, objective: NodeId, wrt: NodeId) -> NodeId {
        match self.node_by_id(wrt) {
            NodeDef::Symbol(_) => {}
            def => panic!("Cannot take gradient w.r.t non-symbol {:?}", def),
        };

        if wrt == objective {
            return self.node(NodeDef::Constant(1.0.into()));
        } else if !self.contains(wrt, objective) {
            return self.node(NodeDef::Constant(0.0.into()));
        }

        match self.node_by_id(objective) {
            NodeDef::Symbol(_) |
            NodeDef::Constant(_) => unreachable!(),

            NodeDef::Unary(op, of) => {
                let subgradient = self.gradient(of, wrt);
                match op {
                    Unary::Neg => self.node(NodeDef::Unary(Unary::Neg, subgradient)),
                    Unary::Sin => {
                        self.node(NodeDef::Binary(Binary::Mul,
                                                  self.node(NodeDef::Unary(Unary::Cos, of)),
                                                  subgradient))
                    }
                    Unary::Cos => {
                            self.node(
                                NodeDef::Unary(
                                    Unary::Neg,
                                    self.node(
                                        NodeDef::Binary(
                                            Binary::Mul,
                                            self.node(NodeDef::Unary(Unary::Sin, of)),
                                            subgradient))))
                        }
                    Unary::Log => self.node(NodeDef::Binary(Binary::Div, subgradient, of)),
                    Unary::Exp => self.node(NodeDef::Binary(Binary::Mul, objective, subgradient)),
                }
            }

            NodeDef::Binary(op, lhs, rhs) => {
                let lhs_contains = self.contains(wrt, lhs);
                let rhs_contains = self.contains(wrt, rhs);

                let lhs_gradient = self.gradient(lhs, wrt);
                let rhs_gradient = self.gradient(rhs, wrt);

                if !lhs_contains {
                    match op {
                        Binary::Add => rhs_gradient,
                        Binary::Sub => self.node(NodeDef::Unary(Unary::Neg, rhs_gradient)),
                        Binary::Mul => self.node(NodeDef::Binary(Binary::Mul, lhs, rhs_gradient)),
                        Binary::Div => self.node(
                            NodeDef::Binary(
                                Binary::Div,
                                self.node(NodeDef::Binary(Binary::Mul, lhs, rhs_gradient)),
                                self.node(
                                    NodeDef::Binary(
                                        Binary::Pow,
                                        rhs,
                                        self.node(NodeDef::Constant(2.0.into())))))),
                        Binary::Pow => self.node(
                            NodeDef::Binary(
                                Binary::Mul,
                                objective,
                                self.node(
                                    NodeDef::Binary(
                                        Binary::Mul,
                                        self.node(NodeDef::Unary(Unary::Log, lhs)),
                                        rhs_gradient)))),
                    }
                } else if !rhs_contains {
                    match op {
                        Binary::Add | Binary::Sub => lhs_gradient,
                        Binary::Mul => self.node(NodeDef::Binary(Binary::Mul, lhs_gradient, rhs)),
                        Binary::Div => self.node(NodeDef::Binary(Binary::Div, lhs_gradient, rhs)),
                        Binary::Pow => self.node(
                            NodeDef::Binary(
                                Binary::Mul,
                                self.node(
                                    NodeDef::Binary(
                                        Binary::Pow,
                                        lhs,
                                        self.node(
                                            NodeDef::Binary(
                                                Binary::Sub,
                                                rhs,
                                                self.node(NodeDef::Constant(1.0.into())))))),
                                                self.node(NodeDef::Binary(
                                                        Binary::Mul, rhs, lhs_gradient)))),
                    }
                } else {
                    match op {
                        Binary::Add => {
                            self.node(NodeDef::Binary(Binary::Add, lhs_gradient, rhs_gradient))
                        }
                        Binary::Sub => {
                            self.node(NodeDef::Binary(Binary::Sub, lhs_gradient, rhs_gradient))
                        }
                        Binary::Mul => {
                            self.node(NodeDef::Binary(Binary::Add,
                                                      self.node(NodeDef::Binary(Binary::Mul,
                                                                                lhs,
                                                                                rhs_gradient)),
                                                      self.node(NodeDef::Binary(Binary::Mul,
                                                                                lhs_gradient,
                                                                                rhs))))
                        }
                        Binary::Div => self.node(
                            NodeDef::Binary(
                                Binary::Div,
                                self.node(
                                    NodeDef::Binary(
                                        Binary::Sub,
                                        self.node(
                                            NodeDef::Binary(Binary::Mul, lhs, rhs_gradient)),
                                            self.node(
                                                NodeDef::Binary(Binary::Mul, lhs_gradient, rhs)))),
                                                self.node(
                                                    NodeDef::Binary(
                                                        Binary::Pow,
                                                        rhs,
                    self.node(NodeDef::Constant(0.0.into())))))),
                        Binary::Pow => self.gradient(
                            self.node(
                                NodeDef::Unary(
                                    Unary::Exp,
                                    self.node(
                                        NodeDef::Binary(
                                            Binary::Mul,
                                            self.node(NodeDef::Unary(Unary::Log, lhs)),
                                            rhs)))),
                                            wrt),

                    }
                }
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct Graph {
    inner: Rc<GraphData>,
}

impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    pub fn scalar(&self, name: &str) -> Node {
        self.symbol(SymbolDef { name: name.to_owned() })
    }

    pub fn constant(&self, value: f32) -> Node {
        self.node(NodeDef::Constant(OrderedFloat(value)))
    }

    pub fn gradient(&self, objective: &Node, wrt: &Node) -> Node {
        Node {
            id: self.inner.gradient(objective.id, wrt.id),
            graph: self.clone(),
        }
    }

    fn node_by_id(&self, id: NodeId) -> NodeDef {
        *self.inner.id_to_node.borrow().get(&id).expect("")
    }

    fn symbol_by_id(&self, id: SymbolId) -> SymbolDef {
        self.inner.id_to_symbol.borrow().get(&id).expect("").clone()
    }

    fn symbol(&self, def: SymbolDef) -> Node {
        Node {
            id: self.inner.symbol(def),
            graph: self.clone(),
        }
    }

    fn node(&self, def: NodeDef) -> Node {
        Node {
            id: self.inner.node(def),
            graph: self.clone(),
        }
    }

    fn dump_program(&self, outputs: &[Node], inline: Inline) {
        let mut expand = Vec::new();
        let mut id_to_level = HashMap::new();
        let mut use_count = HashMap::new();
        expand.extend(outputs.iter().map(|n| n.id));
        while let Some(node_id) = expand.pop() {
            if id_to_level.contains_key(&node_id) {
                continue;
            }

            match self.node_by_id(node_id) {
                NodeDef::Unary(_, of_id) => {
                    if let Some(child_level) = id_to_level.get(&of_id).cloned() {
                        *use_count.entry(of_id).or_insert(0) += 1;
                        id_to_level.insert(node_id, child_level + 1);
                    } else {
                        expand.push(node_id);
                        expand.push(of_id);
                    }
                }
                NodeDef::Binary(_, lhs_id, rhs_id) => {
                    match (id_to_level.get(&lhs_id).cloned(), id_to_level.get(&rhs_id).cloned()) {
                        (Some(lhs_level), Some(rhs_level)) => {
                            *use_count.entry(lhs_id).or_insert(0) += 1;
                            *use_count.entry(rhs_id).or_insert(0) += 1;
                            id_to_level.insert(node_id, cmp::max(lhs_level, rhs_level) + 1);
                        }
                        (lhs_level, rhs_level) => {
                            expand.push(node_id);
                            if lhs_level.is_none() {
                                expand.push(lhs_id);
                            }
                            if rhs_level.is_none() {
                                expand.push(rhs_id);
                            }
                        }
                    }
                }
                NodeDef::Constant(..) |
                NodeDef::Symbol(..) => {
                    id_to_level.insert(node_id, 0);
                }
            }
        }
        let mut show: HashMap<NodeId, String> = HashMap::new();
        let mut sorted = id_to_level.into_iter()
            .map(|(id, level)| (level, id))
            .collect::<Vec<_>>();
        sorted.sort();

        for (level, id) in sorted.into_iter() {
            let node = Node {
                id: id,
                graph: self.clone(),
            };

            let def = node.def();
            let value = match def {
                NodeDef::Constant(value) => format!("{}", value),
                NodeDef::Symbol(symbol_id) => self.symbol_by_id(symbol_id).name.clone(),
                NodeDef::Unary(op, of_id) => op.show(&show[&of_id]),
                NodeDef::Binary(op, lhs_id, rhs_id) => op.show(&show[&lhs_id], &show[&rhs_id]),
            };

            if let Some(pos) = outputs.iter().position(|n| n.id == id) {
                println!("output_{} <- {} [{}]", pos, value, level);
            } else if use_count[&id] == 1 && inline == Inline::Yes {
                show.insert(id, value);
            } else {
                let name = node.name();
                show.insert(id, name.clone());
                match def {
                    NodeDef::Constant(..) |
                    NodeDef::Symbol(..) => {}
                    NodeDef::Unary(..) |
                    NodeDef::Binary(..) => {
                        println!("{} <- {} [{}]", name, value, level);
                    }
                }
            }
        }
    }
}


#[derive(Clone)]
pub struct SymbolDef {
    name: String,
}

#[derive(Clone)]
pub struct Node {
    graph: Graph,
    id: NodeId,
}

impl Node {
    pub fn name(&self) -> String {
        let prefix;
        match self.def() {
            NodeDef::Constant(f) => return format!("{}", f),
            NodeDef::Symbol(symbol_id) => return self.graph.symbol_by_id(symbol_id).name.clone(),
            NodeDef::Unary(op, _) => {
                prefix = match op {
                    Unary::Neg => "neg",
                    Unary::Sin => "sin",
                    Unary::Cos => "cos",
                    Unary::Log => "log",
                    Unary::Exp => "exp",
                };
            }
            NodeDef::Binary(op, _, _) => {
                prefix = match op {
                    Binary::Add => "add",
                    Binary::Sub => "sub",
                    Binary::Mul => "mul",
                    Binary::Div => "div",
                    Binary::Pow => "pow",
                };
            }
        }
        format!("{}_{}", prefix, self.id.0)
    }
}

impl Node {
    pub fn def(&self) -> NodeDef {
        self.graph.node_by_id(self.id)
    }

    pub fn sin(&self) -> Node {
        self.graph.node(NodeDef::Unary(Unary::Sin, self.id))
    }

    pub fn cos(&self) -> Node {
        self.graph.node(NodeDef::Unary(Unary::Cos, self.id))
    }

    pub fn log(&self) -> Node {
        self.graph.node(NodeDef::Unary(Unary::Log, self.id))
    }

    pub fn exp(&self) -> Node {
        self.graph.node(NodeDef::Unary(Unary::Exp, self.id))
    }

    pub fn pow(&self, rhs: &Node) -> Node {
        self.graph.node(NodeDef::Binary(Binary::Pow, self.id, rhs.id))
    }

    pub fn gradient(&self, wrt: &Node) -> Node {
        self.graph.gradient(self, wrt)
    }
}

macro_rules! impl_binary_op {
    ($([$op:ident, $fun:ident])+) => {
        $(
            impl<'a> $op <&'a Node> for &'a Node {
                type Output = Node;

                fn $fun(self, rhs: &'a Node) -> Node {
                    self.graph.node(NodeDef::Binary(Binary::$op, self.id, rhs.id))
                }
            }

            impl<'a> $op <Node> for &'a Node {
                type Output = Node;

                fn $fun(self, rhs: Node) -> Node {
                    $op::$fun(self, &rhs)
                }
            }

            impl<'a> $op <&'a Node> for Node {
                type Output = Node;

                fn $fun(self, rhs: &'a Node) -> Node {
                    $op::$fun(&self, rhs)
                }
            }

            impl $op <Node> for Node {
                type Output = Node;

                fn $fun(self, rhs: Node) -> Node {
                    $op::$fun(&self, &rhs)
                }
            }

            impl<'a> $op <f32> for &'a Node {
                type Output = Node;

                fn $fun(self, rhs: f32) -> Node {
                    $op::$fun(self, self.graph.node(NodeDef::Constant(rhs.into())))
                }
            }

            impl<'a> $op <f32> for Node {
                type Output = Node;

                fn $fun(self, rhs: f32) -> Node {
                    $op::$fun(&self, rhs)
                }
            }

            impl<'a> $op <&'a Node> for f32 {
                type Output = Node;

                fn $fun(self, rhs: &'a Node) -> Node {
                    $op::$fun(rhs.graph.node(NodeDef::Constant(self.into())), rhs)
                }
            }

            impl<'a> $op <Node> for f32 {
                type Output = Node;

                fn $fun(self, rhs: Node) -> Node {
                    $op::$fun(self, &rhs)
                }
            }
            )+
    }
}

macro_rules! impl_unary_op {
    ($([$op:ident, $fun:ident])+) => {
        $(
            impl<'a> $op for &'a Node {
                type Output = Node;

                fn $fun(self) -> Node {
                    self.graph.node(NodeDef::Unary(Unary::$op, self.id))
                }
            }

            impl $op for Node {
                type Output = Node;

                fn $fun(self) -> Node {
                    $op::$fun(&self)
                }
            }
         )+
    }
}

impl_binary_op! {
    [Add, add]
        [Sub, sub]
            [Mul, mul]
                [Div, div]
}

impl_unary_op! {
    [Neg, neg]
}


#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl Binary {
    pub fn show(self, left: &str, right: &str) -> String {
        match self {
            Binary::Add => format!("({} + {})", left, right),
            Binary::Sub => format!("({} - {})", left, right),
            Binary::Mul => format!("({} * {})", left, right),
            Binary::Div => format!("({} / {})", left, right),
            Binary::Pow => format!("pow({}, {})", left, right),
        }
    }

    pub fn eval(self, left: f32, right: f32) -> f32 {
        match self {
            Binary::Add => left + right,
            Binary::Sub => left - right,
            Binary::Mul => left * right,
            Binary::Pow => left.powf(right),
            Binary::Div => left / right,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum Unary {
    Neg,
    Sin,
    Cos,
    Log,
    Exp,
}

impl Unary {
    pub fn show(self, of: &str) -> String {
        match self {
            Unary::Neg => format!("-{}", of),
            Unary::Sin => format!("sin({})", of),
            Unary::Cos => format!("cos({})", of),
            Unary::Log => format!("log({})", of),
            Unary::Exp => format!("exp({})", of),
        }
    }

    pub fn eval(self, of: f32) -> f32 {
        match self {
            Unary::Neg => -of,
            Unary::Sin => of.sin(),
            Unary::Cos => of.cos(),
            Unary::Log => of.ln(),
            Unary::Exp => of.exp(),
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum NodeDef {
    Constant(OrderedFloat<f32>),
    Symbol(SymbolId),
    Binary(Binary, NodeId, NodeId),
    Unary(Unary, NodeId),
}
