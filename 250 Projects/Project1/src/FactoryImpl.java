import java.util.HashSet;
import java.util.NoSuchElementException;

public class FactoryImpl implements Factory{
    private Holder first;
    private Holder last;
    private Integer size = 0;


    /**
     * Inserts {@code product} at the beginning of this factory line.
     *
     * @param product the product to add
     */
    @Override
    public void addFirst(Product product) {
        Holder holder;
        if (size==0){
            holder = new Holder(null, product, null);
            last = holder;
        }
        else{
            holder = new Holder(null, product, first);
            first.setPreviousHolder(holder);
        }
        first = holder;
        size++;
    }

    /**
     * Inserts {@code product} to the end of this factory line.
     *
     * @param product the product to add
     */
    @Override
    public void addLast(Product product) {
        Holder holder;
        if(size==0){
            holder = new Holder(null, product, null);
            first = holder;
        }
        else{
            holder = new Holder(last, product, null);
            last.setNextHolder(holder);
        }
        last = holder;
        size++;
    }

    /**
     * Insert the {@code product} at the specified position in this factory line.
     * Shifts the products currently at and to the right of that position.
     *
     * @param index   index at which the product is to be inserted
     * @param product the product to be inserted
     * @throws IndexOutOfBoundsException {@inheritDoc}
     */
    @Override
    public void add(int index, Product product) throws IndexOutOfBoundsException {
        if(index > size || index < 0){
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        if(index==0){
            addFirst(product);
        }
        else if(index==size){
            addLast(product);
        }
        else{
            if (index <= size/2){
                Holder iter = first;
                for (int i = 0; i<index; i++){
                    iter = iter.getNextHolder();
                }
                Holder prev = iter.getPreviousHolder();
                Holder holder = new Holder(prev,product,iter);
                prev.setNextHolder(holder);
                iter.setPreviousHolder(holder);
            }
            else{
                Holder iter = last;
                for (int i = size ; i>index; i--){
                    iter = iter.getPreviousHolder();
                }
                Holder next = iter.getNextHolder();
                Holder holder = new Holder(iter, product, next);
                next.setPreviousHolder(holder);
                iter.setNextHolder(holder);
            }
            size++;
        }
    }

    /**
     * Removes and returns the first product from this factory line.
     *
     * @return the first product from this factory line
     * @throws NoSuchElementException if the line is empty
     */
    @Override
    public Product removeFirst() throws NoSuchElementException {
        if (size==0){
            throw new NoSuchElementException("Factory is empty.");
        }
        else{
            Holder removed = first;
            Product removedProduct = removed.getProduct();
            removed.setProduct(null);
            if(size==1){
                first = null;
                last = null;
            }
            else{
                first = removed.getNextHolder();
                first.setPreviousHolder(null);
                removed.setNextHolder(null);
            }
            size--;
            return removedProduct;
        }
    }

    /**
     * Removes and returns the last product from this factory line.
     *
     * @return the last product from this factory line
     * @throws NoSuchElementException if the line is empty
     */
    @Override
    public Product removeLast() throws NoSuchElementException {
        if (size==0){
            throw new NoSuchElementException("Factory is empty.");
        }
        else{
            Holder removed = last;
            Product removedProduct = removed.getProduct();
            removed.setProduct(null);
            if(size==1){
                first = null;
                last = null;
            }
            else{
                last = removed.getPreviousHolder();
                last.setNextHolder(null);
                removed.setPreviousHolder(null);
            }
            size--;
            return removedProduct;
        }
    }

    /**
     * Finds and returns the product with the specified {@code id}.
     *
     * @param id id of the product
     * @return the product with the specified id
     * @throws NoSuchElementException if the product does not exist
     */
    @Override
    public Product find(int id) throws NoSuchElementException {
        Holder iter = first;
        while(iter!=null){
            if(iter.getProduct().getId()== id){
                return iter.getProduct();
            }
            iter = iter.getNextHolder();
        }
        throw new NoSuchElementException("Product not found.");
    }

    /**
     * Updates the product value with {@code id} in this factory line
     * with the given {@code newValue}.
     *
     * @param id    id of the product to update
     * @param value new value for the specified product
     * @return the previous product with {@code id}
     * @throws NoSuchElementException if the product does not exist
     */
    @Override
    public Product update(int id, Integer value) throws NoSuchElementException {
        Product product;
        product = find(id);
        int prevValue = product.getValue();
        product.setValue(value);
        return new Product(id, prevValue);
    }

    /**
     * Returns the product at the specified position in this factory line.
     *
     * @param index index of the product to return
     * @return the product at the specified position in this factory line
     * @throws IndexOutOfBoundsException {@inheritDoc}
     */
    @Override
    public Product get(int index) throws IndexOutOfBoundsException {
        if (index>=size || index < 0){
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        else{
            Holder iter = getHolder(index);
            return iter.getProduct();
        }

    }

    /**
     * Removes and returns the product at the specified position in this
     * factory line.
     *
     * @param index index of the product to remove
     * @return the removed {@code product}
     * @throws IndexOutOfBoundsException {@inheritDoc}
     */
    @Override
    public Product removeIndex(int index) throws IndexOutOfBoundsException {
        if (index>=size || index < 0){
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        else if(index==0){
            return removeFirst();
        }
        else if(index==size-1){
            return removeLast();
        }
        else{
            Holder iter = getHolder(index);
            return removeFromInside(iter);
        }
    }

    // Returns holder at the specified index
    private Holder getHolder(int index) {
        Holder iter;
        if(index <=size/2){
            iter = first;
            for (int i = 0; i< index; i++){
                iter = iter.getNextHolder();
            }
        }
        else{
            iter = last;
            for (int i = size - 1; i> index; i--){
                iter = iter.getPreviousHolder();
            }
        }
        return iter;
    }

    /**
     * Removes the first occurrence of the {@code product} with the
     * specified {@code value} from this factory line. If this factory
     * line does not contain a product with the specified {@code value},
     * it is unchanged. More formally, removes the {@code product} with the
     * lowest index {@code i} such that {@code product.getValue() == value}.
     *
     * @param value value of the {@code product} to be removed from
     *              this factory line, if present
     * @return the removed {@code product}
     * @throws NoSuchElementException if the product with the given
     *                                {@code value} does not exist
     */
    @Override
    public Product removeProduct(int value) throws NoSuchElementException {
        Holder iter = first;
        while(iter!=null){
            if(iter.getProduct().getValue()==value){
                if(first.equals(iter)){
                    return removeFirst();
                }
                else if(last.equals(iter)){
                    return removeLast();
                }
                else{
                    return removeFromInside(iter);
                }
            }
            iter = iter.getNextHolder();
        }
        throw new NoSuchElementException("Product not found.");
    }

    // Removes given holder that is not in any end points
    private Product removeFromInside(Holder iter) {
        Holder next = iter.getNextHolder();
        Holder prev= iter.getPreviousHolder();
        prev.setNextHolder(next);
        next.setPreviousHolder(prev);
        Product removedProduct = iter.getProduct();
        iter.setProduct(null);
        iter.setPreviousHolder(null);
        iter.setNextHolder(null);
        size--;
        return removedProduct;
    }

    /**
     * Filters the factory line such that every duplicate product is removed.
     * Duplicate products are products with the same value in this context.
     *
     * @return number of removed products
     */
    @Override
    public int filterDuplicates() {
        HashSet<Integer> values = new HashSet<>();
        Holder iter = first;
        int count = 0;
        while(iter!=null){
            Holder temp = iter;
            iter = iter.getNextHolder();
            if(values.contains(temp.getProduct().getValue())){
                if (temp==last){
                    removeLast();
                }
                else{
                    removeFromInside(temp);
                }
                count++;
            }
            else{
                values.add(temp.getProduct().getValue());
            }
        }
        return count;
    }

    /**
     * Reverses the factory line.
     */
    @Override
    public void reverse() {
        Holder iter = first;
        Holder prev_first = first;
        Holder prev_last = last;
        Holder prev =null;
        while(iter!=null){
            Holder a = iter;
            iter = iter.getNextHolder();
            a.setPreviousHolder(iter);
            a.setNextHolder(prev);
            prev = a;
        }
        last = prev_first;
        first = prev_last;
    }

    /**
     * Prints the factory line
     */
    public void print(){
        Holder iter = first;
        System.out.print("{");
        while(iter!=null){
            System.out.print(iter);
            iter = iter.getNextHolder();
            if (iter!=null){
                System.out.print(",");
            }
        }
        System.out.println("}");
    }
}