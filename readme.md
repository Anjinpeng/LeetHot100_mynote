## LeetCode Hot 100

#### Hash:

##### 1.两数之和：

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。

你可以按任意顺序返回答案。

**示例 1：**

```
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
```

**示例 2：**

```
输入：nums = [3,2,4], target = 6
输出：[1,2]
```

**示例 3：**

```
输入：nums = [3,3], target = 6
输出：[0,1]
```

**思路：**可以用字典作为哈希表来进行查找，因为可以通过in, not in 实现O(1)查找。这里让返回序号，所以可以用num作为key,  idx作为value。遍历列表nums,若target-num在哈希表中,则返回当前的idx和hash[target-num]这两个数字的idx。

题解：

```python
def twoSum(self, nums: List[int], target: int) -> List[int]:
        hs={}
        for idx, num in enumerate(nums):
            if target - num not in hs :
                hs[num]=idx
            else:
                return [hs[target-num],idx]
        return []
```

#### 49.字母异位词分组

给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

 

**示例 1:**

**输入:** strs = ["eat", "tea", "tan", "ate", "nat", "bat"]

**输出:** [["bat"],["nat","tan"],["ate","eat","tea"]]

**解释：**

- 在 strs 中没有字符串可以通过重新排列来形成 `"bat"`。
- 字符串 `"nat"` 和 `"tan"` 是字母异位词，因为它们可以重新排列以形成彼此。
- 字符串 `"ate"` ，`"eat"` 和 `"tea"` 是字母异位词，因为它们可以重新排列以形成彼此。

**示例 2:**

**输入:** strs = [""]

**输出:** [[""]]

**示例 3:**

**输入:** strs = ["a"]

**输出:** [["a"]]

 **思路：**按照某个共同的标准分组，此时可以用哈希表，字典即可，共同特征作为keys，相同特征的字符串组成列表作为values。这里特征就是拥有共同的字母只不过顺序不同，排序后相同。注意，sorted(str)返回的是一个列表。dict.keys(), dict.values(), dict.items()返回的都是迭代对象。

**题解：**

```
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hs={}
        for st in strs:
            st_d="".join(sorted(st))
            if st_d in hs:
                hs[st_d].append(st)
            else :
                hs[st_d]=[st]
        return list(hs.values())
```

#### 128.最长连续序列

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

 

**示例 1：**

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

**示例 2：**

```
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```

**示例 3：**

```
输入：nums = [1,0,1,2]
输出：3
```

思路：这道题主要限制复杂度O(n),所以不能排序，可以用集合作为哈希表，这样对于每个num，都可以O(1)查找num+1或者num-1在不在集合中。遍历集合，即可实现O(n)查找最大连续序列。

题解：

```
def longestConsecutive(self, nums: List[int]) -> int:
        sts=set(nums)
        ans=0
        for st in sts:
            if st-1 in sts:
                continue
            y=st+1
            while y in sts:
                y+=1
            ans=max(ans , y-st)
        return ans
```

#### 二叉树

#### 94.二叉树的中序遍历

给定一个二叉树的根节点 `root` ，返回 *它的 **中序** 遍历* 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```
输入：root = [1,null,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：root = []
输出：[]
```

**示例 3：**

```
输入：root = [1]
输出：[1]
```

#### 思路：

利用递归的思想，先遍历左子树，再是根，最后是右子树，两个子树递归调用中序遍历函数即可。

#### 题解：

```
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

#### 98.验证二叉搜索树

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **严格小于** 当前节点的数。
- 节点的右子树只包含 **严格大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/12/01/tree1.jpg)

```
输入：root = [2,1,3]
输出：true
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/12/01/tree2.jpg)

```
输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5 ，但是右子节点的值是 4 
```

**思路：**验证二叉搜索树要保证左子树小于根小于右子树，那么对于中序遍历来说，就是一个严格递增的列表。如果弄不明白递归时比较大小，那么就得到列表后再比较也没关系。

**题解：**

```
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def midsearch(root):
            if not root:
                return []
            return midsearch(root.left)+[root.val]+midsearch(root.right)
        res=midsearch(root)
        res_num=len(res)
        if res_num<=1:
            return True
        for i in range(1,res_num):
            if res[i]<=res[i-1]:
                return False
        return True 
```

#### 108.将有序数组变成二叉搜索树

给你一个整数数组 `nums` ，其中元素已经按 **升序** 排列，请你将其转换为一棵 平衡 二叉搜索树。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/02/18/btree1.jpg)

```
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/02/18/btree.jpg)

```
输入：nums = [1,3]
输出：[3,1]
解释：[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。
```

**思路：**分治的思想，由于数组已经排好序了，所以用中位数作为根节点，左边切片递归生成左子树，右边切片递归生成右子树。递归结束条件就是列表为空，返回None。

**题解：**

```
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        len_nums= len(nums)
        if len_nums==1:
            root=TreeNode(nums[0])
            return root
        elif len_nums==0:
            return None
        mid = len_nums//2
        root=TreeNode(val=nums[mid])
        root.left=self.sortedArrayToBST(nums[:mid])
        if mid<len_nums:
            root.right=self.sortedArrayToBST(nums[mid+1:])
        return root
```



#### 104.二叉树的最大深度

给定一个二叉树 `root` ，返回其最大深度。

二叉树的 **最大深度** 是指从根节点到最远叶子节点的最长路径上的节点数。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)

 

```
输入：root = [3,9,20,null,null,15,7]
输出：3
```

**示例 2：**

```
输入：root = [1,null,2]
输出：2
```

#### 思路：

利用递归的思想，计算左右子树的最大深度，取最大值+1就是目前的最大深度。

#### 题解：

```
def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1
```

#### 226.翻转二叉树

给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)

```
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/03/14/invert2-tree.jpg)

```
输入：root = [2,1,3]
输出：[2,3,1]
```

#### 思路：

像这种每个子树都做出相同的改变，这里是翻转，所以就可以利用递归的思想直接得到。

#### 题解：

```
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        
        temp=self.invertTree(root.right)
        root.right=self.invertTree(root.left)
        root.left=temp
        return root
```

#### 102.二叉树的层序遍历

给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg)

```
输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
```

**示例 2：**

```
输入：root = [1]
输出：[[1]]
```

**示例 3：**

```
输入：root = []
输出：[]
```

#### 思路：

层序遍历可以利用队列的数据结构，先把根放进队列，然后依次把左子树和右子树放进队列，先进先出即可层序遍历。值得注意的是，每层需要一个列表，所以加一个for循环用temp列表来记录。注意，temp=[],不要放到for循环里面。

#### 题解：

```
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res=[]
        queue=[root]
        while queue:
            temp=[]
            for _ in range(len(queue)):
                node=queue.pop(0)
                temp.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(temp)
        return res
```

#### 199.二叉树的右视图

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

 

**示例 1：**

**输入：**root = [1,2,3,null,5,null,4]

**输出：**[1,3,4]

**解释：**

![img](https://assets.leetcode.com/uploads/2024/11/24/tmpd5jn43fs-1.png)

**示例 2：**

**输入：**root = [1,2,3,4,null,null,null,5]

**输出：**[1,3,4,5]

**解释：**

![img](https://assets.leetcode.com/uploads/2024/11/24/tmpkpe40xeh-1.png)

**示例 3：**

**输入：**root = [1,null,3]

**输出：**[1,3]

**示例 4：**

**输入：**root = []

**输出：**[]

**思路：**所谓的右视图就是每层的最后一个节点，所以利用上题层序遍历，提取每层的最后一个节点即可。

**题解：**

```
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        def levelsearch(root):
            if not root :
                return []
            res=[]
            queue=[root]
            while queue :
                temp=[]
                for _ in range(len(queue)):
                    node=queue.pop(0)
                    temp.append(node.val)
                    if node.left :
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
                res.append(temp)
            return res
        if not root:
            return []
        res=levelsearch(root)
        ans=[]
       
        for ls in res:
            ans.append(ls[-1])
        return ans
```

#### 贪心算法

#### 121.买卖股票的最佳时机

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

 

**示例 1：**

```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

**示例 2：**

```
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
```

**思路：**从1开始遍历列表，对于每一个num,计算这个num和之前最小的数字min_num的差，也就是这一天卖出的最大值，和之前的最大值比较，取最大的作为结果，同时将num加入切片，更新切片最小值。遍历完列表后，相当于比较了所有天数卖出的最大收益。

**题解：**

```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res=0
        min_num=prices[0]
        for i in range(1,len(prices)):
            res=max(res,prices[i]-min_num)
            min_num=min(prices[i],min_num)
        return res
```

#### 55.跳跃游戏

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

**示例 2：**

```
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```

**思路：**从位置0开始维护一个可以到达的最远位置，遍历0和最远位置间的数，更新最远位置，如果最后可以达到最后一个数则为真，否则为假，这里注意用while循环的结束条件。每次i+1，但是i不能超过最大位置。

**题解：**

```
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_step=0
        max_idx=len(nums)-1
        i=0
        while i<=max_step and i <=max_idx:
            max_step=max(max_step,i+nums[i])
            i+=1
        return max_step>=max_idx
```

#### 45.跳跃游戏2

给定一个长度为 `n` 的 **0 索引**整数数组 `nums`。初始位置在下标 0。

每个元素 `nums[i]` 表示从索引 `i` 向后跳转的最大长度。换句话说，如果你在索引 `i` 处，你可以跳转到任意 `(i + j)` 处：

- `0 <= j <= nums[i]` 且
- `i + j < n`

返回到达 `n - 1` 的最小跳跃次数。测试用例保证可以到达 `n - 1`。

 

**示例 1:**

```
输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

**示例 2:**

```
输入: nums = [2,3,0,1,4]
输出: 2
```

 **思路：**这里是都能到达最后，求最小步数，最直接的思路就是计算一下每一步可以走到哪里最远，那么第一次到达最后位置就是最小步数。这里维护一个left和right来表示每一步可以走的范围，每一步用for循环算出下一步的最远距离right,那么left就是上一步的right+1。一旦right可以到达最后位置，结束while循环.

**题解：**

```
class Solution:
    def jump(self, nums: List[int]) -> int:
        res=0
        left=0
        right=0
        len_num=len(nums)-1
        while right<len_num:
            res+=1
            temp=0
            for i in range(left,right+1):
                temp=max(temp,nums[i]+i)
            left=right+1
            right=temp
        return res
```

#### 763.划分字母区间

给你一个字符串 `s` 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。例如，字符串 `"ababcc"` 能够被分为 `["abab", "cc"]`，但类似 `["aba", "bcc"]` 或 `["ab", "ab", "cc"]` 的划分是非法的。

注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 `s` 。

返回一个表示每个字符串片段的长度的列表。

 

**示例 1：**

```
输入：s = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca"、"defegde"、"hijhklij" 。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 这样的划分是错误的，因为划分的片段数较少。 
```

**示例 2：**

```
输入：s = "eccbbbbdec"
输出：[10]
```

 **思路：**遍历整个字符串，初始化一个right为该与首字母相同的字母的最后一个位置，遍历0到最大位置的所有字母，并更新最大位置，如果更新不动，即idx==right，说明一个片段结束，加入result,并且把计数恢复为0，直接用计数，可以不用维护一个left啦。重点是怎么找到每个字母的最后一个位置，可以用字典保存。

**题解：**

```
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        dic={a:idx for idx,a in enumerate(s)}
        res=[]
        num=0
        right=dic[s[0]]
        for i in range(len(s)):
            num+=1
            if dic[s[i]]>right:
                right=dic[s[i]]
            if i==right:
                res.append(num)
                num=0
        return res
```

