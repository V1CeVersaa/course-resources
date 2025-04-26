// TODO: Fix the compiler error in this function.
fn fill_vec(vec: Vec<i32>) -> Vec<i32> {
    let mut vec = vec;

    vec.push(88);

    vec
}

fn main() {
    // You can optionally experiment here.
    let x = 1;
    let mut y = x;
    y = y + 1;
    println!("y: {}", y);
    println!("x: {}", x);
    let s1 = String::from("hello");
    let mut s2 = s1;
    s2.push_str(", world");
    println!("s2: {}", s2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_semantics1() {
        let vec0 = vec![22, 44, 66];
        let vec1 = fill_vec(vec0);
        assert_eq!(vec1, vec![22, 44, 66, 88]);
    }
}
